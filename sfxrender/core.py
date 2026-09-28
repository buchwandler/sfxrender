"""Renderer registry and public orchestration API."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from copy import deepcopy
from numbers import Integral
from typing import Any

import numpy as np

from .catalog import build_llm_catalog, validate_effect_descriptor
from .catalog import catalog as builtin_catalog
from .errors import InvalidEffectParameterError, SFXRenderError, UnknownEffectError
from .procedural import (
    door_close,
    door_open,
    footsteps,
    knock,
    pen_write,
    phone_ring,
    doorbell_ring,
    switch_toggle,
    button_press,
    object_set_down,
    glass_clink,
    paper_page_turn,
    paper_handle,
    printer_power_switch,
    printer_print,
    printer_restart,
    printer_tray_open,
    printer_tray_close,
    printer_wake,
)
from .types import RenderContext, RenderedSound, SfxSpec
from .uri import parse_sfx_uri

EffectRenderer = Callable[[SfxSpec, RenderContext], RenderedSound]


def _parameter_error(effect: str, name: str, message: str) -> InvalidEffectParameterError:
    return InvalidEffectParameterError(
        f"Invalid parameter {name!r} for effect {effect!r}: {message}"
    )


def _validate_parameters(spec: SfxSpec, descriptor: Mapping[str, Any] | None) -> None:
    """Validate supplied URI fields against an optional effect descriptor."""
    if descriptor is None:
        return
    schemas: Mapping[str, Any] = descriptor["parameters"]
    for name, schema in schemas.items():
        if schema.get("required", False) and name not in spec.parameters:
            raise _parameter_error(spec.effect, name, "parameter is required")

    for name in sorted(spec.parameters):
        if name not in schemas:
            raise _parameter_error(spec.effect, name, "unknown parameter")
        raw = spec.parameters[name]
        schema = schemas[name]
        kind = schema["type"]
        if name == "seed" and kind == "integer":
            try:
                seed = int(raw)
            except (TypeError, ValueError) as exc:
                raise _parameter_error(
                    spec.effect, name, f"expected a non-negative integer, got {raw!r}"
                ) from exc
            if seed < 0:
                raise _parameter_error(
                    spec.effect, name, f"expected a non-negative integer, got {raw!r}"
                )
            continue
        if kind == "enum":
            enum_value = raw.lower()
            allowed = schema["values"]
            if enum_value not in [choice.lower() for choice in allowed]:
                choices = ", ".join(allowed)
                raise _parameter_error(spec.effect, name, f"expected one of {choices}, got {raw!r}")
            continue
        if kind == "integer":
            try:
                numeric_value: int | float = int(raw)
            except (TypeError, ValueError) as exc:
                raise _parameter_error(
                    spec.effect, name, f"{name} must be an integer, got {raw!r}"
                ) from exc
        elif kind in {"number", "seconds"}:
            try:
                numeric_value = float(raw)
            except (TypeError, ValueError) as exc:
                raise _parameter_error(
                    spec.effect, name, f"{name} must be a number, got {raw!r}"
                ) from exc
            if not math.isfinite(numeric_value):
                raise _parameter_error(spec.effect, name, f"expected a finite number, got {raw!r}")
        else:
            raise RuntimeError(f"unsupported catalog parameter type {kind!r}")
        minimum = schema.get("minimum")
        maximum = schema.get("maximum")
        if (minimum is not None and numeric_value < minimum) or (
            maximum is not None and numeric_value > maximum
        ):
            expected = (
                f"{minimum}..{maximum}"
                if minimum is not None and maximum is not None
                else f"at least {minimum}"
                if minimum is not None
                else f"at most {maximum}"
            )
            raise _parameter_error(
                spec.effect, name, f"expected a value in {expected}, got {raw!r}"
            )


def _validate_pcm(samples: object) -> None:
    """Reject renderer results that violate the public PCM contract."""
    if not isinstance(samples, np.ndarray):
        raise SFXRenderError("effect renderer must return samples as a NumPy array")
    if samples.ndim != 1:
        raise SFXRenderError("effect renderer must return mono, one-dimensional PCM")
    if samples.size == 0:
        raise SFXRenderError("effect renderer must return non-empty PCM")
    if samples.dtype != np.float32:
        raise SFXRenderError("effect renderer must return float32 PCM")
    if not np.isfinite(samples).all():
        raise SFXRenderError("effect renderer must return finite PCM values")
    if float(np.max(np.abs(samples))) > 1.0:
        raise SFXRenderError("effect renderer PCM values must be normalized to [-1, 1]")


class SFXRenderer:
    """Resolve semantic SFX specs into deterministic PCM audio."""

    _context: RenderContext
    _renderers: dict[str, EffectRenderer]
    _descriptors: dict[str, dict[str, Any]]

    def __init__(self, *, sample_rate: int = 24_000) -> None:
        if isinstance(sample_rate, bool) or not isinstance(sample_rate, Integral):
            raise TypeError("sample_rate must be an integer")
        sample_rate = int(sample_rate)
        if sample_rate < 8_000:
            raise ValueError("sample_rate must be at least 8000 Hz")
        self._context = RenderContext(sample_rate=sample_rate)
        self._renderers: dict[str, EffectRenderer] = {
            "impact.knock": knock,
            "footsteps.walk": footsteps,
            "phone.ring": phone_ring,
            "doorbell.ring": doorbell_ring,
            "switch.toggle": switch_toggle,
            "button.press": button_press,
            "object.set_down": object_set_down,
            "glass.clink": glass_clink,
            "paper.page_turn": paper_page_turn,
            "paper.handle": paper_handle,
            "door.open": door_open,
            "door.close": door_close,
            "printer.print": printer_print,
            "printer.tray_open": printer_tray_open,
            "printer.tray_close": printer_tray_close,
            "printer.power_switch": printer_power_switch,
            "printer.restart": printer_restart,
            "pen.write": pen_write,
            "printer.wake": printer_wake,
        }
        self._descriptors = builtin_catalog()

    @property
    def sample_rate(self) -> int:
        return self._context.sample_rate

    def effects(self) -> tuple[str, ...]:
        return tuple(sorted(self._renderers))

    def catalog(self) -> dict[str, dict[str, Any]]:
        """Return documented descriptors for effects registered on this renderer."""
        return deepcopy(self._descriptors)

    def llm_catalog(self) -> dict[str, Any]:
        """Return an LLM authoring manifest for this renderer's documented effects."""
        return build_llm_catalog(self._descriptors)

    def register(
        self,
        effect: str,
        renderer: EffectRenderer,
        *,
        descriptor: Mapping[str, Any] | None = None,
        replace: bool = False,
    ) -> None:
        effect = effect.strip()
        if not effect:
            raise ValueError("effect name must not be empty")
        if effect in self._renderers and not replace:
            raise ValueError(f"renderer already registered for {effect!r}")
        normalized_descriptor = (
            validate_effect_descriptor(effect, descriptor) if descriptor is not None else None
        )
        self._renderers[effect] = renderer
        if normalized_descriptor is not None:
            self._descriptors[effect] = normalized_descriptor

    def validate(self, spec: SfxSpec) -> SfxSpec:
        """Validate a request without synthesizing audio and return the same spec."""
        if spec.effect not in self._renderers:
            available = ", ".join(self.effects())
            raise UnknownEffectError(f"unknown SFX effect {spec.effect!r}; available: {available}")
        _validate_parameters(spec, self._descriptors.get(spec.effect))
        return spec

    def validate_uri(self, uri: str) -> SfxSpec:
        """Parse and validate an SFX URI without generating PCM."""
        return self.validate(parse_sfx_uri(uri))

    def render(self, spec: SfxSpec) -> RenderedSound:
        spec = self.validate(spec)
        renderer = self._renderers[spec.effect]
        sound = renderer(spec, self._context)
        if sound.sample_rate != self.sample_rate:
            raise SFXRenderError("effect renderer returned an unexpected sample rate")
        _validate_pcm(sound.samples)
        return sound

    def render_uri(self, uri: str) -> RenderedSound:
        """Parse and render an SFX URI using the public typed-error contract."""
        return self.render(parse_sfx_uri(uri))

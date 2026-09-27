"""Renderer registry and public orchestration API."""

from __future__ import annotations

import math
from collections.abc import Callable
from numbers import Integral
from typing import Any

import numpy as np

from .catalog import catalog
from .errors import InvalidEffectParameterError, SFXRenderError, UnknownEffectError
from .procedural import door_close, door_open, footsteps, knock, phone_ring
from .types import RenderContext, RenderedSound, SfxSpec
from .uri import parse_sfx_uri

EffectRenderer = Callable[[SfxSpec, RenderContext], RenderedSound]


def _parameter_error(effect: str, name: str, message: str) -> InvalidEffectParameterError:
    return InvalidEffectParameterError(
        f"Invalid parameter {name!r} for effect {effect!r}: {message}"
    )


def _validate_builtin_parameters(spec: SfxSpec) -> None:
    """Validate built-in URI fields using the public effect catalog schema."""
    definition = catalog().get(spec.effect)
    if definition is None:
        return
    schemas: dict[str, Any] = definition["parameters"]
    for name in sorted(spec.parameters):
        if name not in schemas:
            raise _parameter_error(spec.effect, name, "unknown parameter")
        raw = spec.parameters[name]
        schema = schemas[name]
        kind = schema["type"]
        if name == "seed":
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
            value = raw.lower()
            allowed = schema["values"]
            if value not in allowed:
                choices = ", ".join(allowed)
                raise _parameter_error(spec.effect, name, f"expected one of {choices}, got {raw!r}")
            continue
        if kind == "integer":
            try:
                value = int(raw)
            except (TypeError, ValueError) as exc:
                raise _parameter_error(
                    spec.effect, name, f"{name} must be an integer, got {raw!r}"
                ) from exc
        elif kind in {"number", "seconds"}:
            try:
                value = float(raw)
            except (TypeError, ValueError) as exc:
                raise _parameter_error(
                    spec.effect, name, f"{name} must be a number, got {raw!r}"
                ) from exc
            if not math.isfinite(value):
                raise _parameter_error(spec.effect, name, f"expected a finite number, got {raw!r}")
        else:
            raise RuntimeError(f"unsupported catalog parameter type {kind!r}")
        minimum = schema.get("minimum")
        maximum = schema.get("maximum")
        if (minimum is not None and value < minimum) or (maximum is not None and value > maximum):
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
            "door.open": door_open,
            "door.close": door_close,
        }

    @property
    def sample_rate(self) -> int:
        return self._context.sample_rate

    def effects(self) -> tuple[str, ...]:
        return tuple(sorted(self._renderers))

    def register(self, effect: str, renderer: EffectRenderer, *, replace: bool = False) -> None:
        effect = effect.strip()
        if not effect:
            raise ValueError("effect name must not be empty")
        if effect in self._renderers and not replace:
            raise ValueError(f"renderer already registered for {effect!r}")
        self._renderers[effect] = renderer

    def render(self, spec: SfxSpec) -> RenderedSound:
        renderer = self._renderers.get(spec.effect)
        if renderer is None:
            available = ", ".join(self.effects())
            raise UnknownEffectError(f"unknown SFX effect {spec.effect!r}; available: {available}")
        _validate_builtin_parameters(spec)
        sound = renderer(spec, self._context)
        if sound.sample_rate != self.sample_rate:
            raise SFXRenderError("effect renderer returned an unexpected sample rate")
        _validate_pcm(sound.samples)
        return sound

    def render_uri(self, uri: str) -> RenderedSound:
        """Parse and render an SFX URI using the public typed-error contract."""
        return self.render(parse_sfx_uri(uri))

"""Renderer registry and public orchestration API."""

from __future__ import annotations

from collections.abc import Callable

from .procedural import door_open, footsteps, knock, phone_ring
from .types import RenderContext, RenderedSound, SfxSpec
from .uri import parse_sfx_uri

EffectRenderer = Callable[[SfxSpec, RenderContext], RenderedSound]


class SFXRenderer:
    """Resolve semantic SFX specs into deterministic PCM audio."""

    def __init__(self, *, sample_rate: int = 24_000) -> None:
        if sample_rate < 8_000:
            raise ValueError("sample_rate must be at least 8000 Hz")
        self._context = RenderContext(sample_rate=sample_rate)
        self._renderers: dict[str, EffectRenderer] = {
            "impact.knock": knock,
            "footsteps.walk": footsteps,
            "phone.ring": phone_ring,
            "door.open": door_open,
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
        try:
            renderer = self._renderers[spec.effect]
        except KeyError as exc:
            available = ", ".join(self.effects())
            raise ValueError(f"unknown SFX effect {spec.effect!r}; available: {available}") from exc
        sound = renderer(spec, self._context)
        if sound.sample_rate != self.sample_rate:
            raise ValueError("effect renderer returned an unexpected sample rate")
        return sound

    def render_uri(self, uri: str) -> RenderedSound:
        return self.render(parse_sfx_uri(uri))

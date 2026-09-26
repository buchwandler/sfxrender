"""Parser for the public ``sfx:`` URI contract."""

from __future__ import annotations

from urllib.parse import parse_qsl, urlsplit

from .types import SfxSpec


class SfxUriError(ValueError):
    """Raised when an SFX URI is malformed."""


def parse_sfx_uri(uri: str) -> SfxSpec:
    """Parse ``sfx:effect.name?key=value`` into an :class:`SfxSpec`."""

    parts = urlsplit(uri)
    if parts.scheme.lower() != "sfx":
        raise SfxUriError("SFX URI must use the 'sfx' scheme")
    if parts.netloc:
        raise SfxUriError("SFX URI must not contain an authority component")
    effect = parts.path.strip()
    if not effect:
        raise SfxUriError("SFX URI must contain an effect name")
    if any(ch.isspace() for ch in effect):
        raise SfxUriError("effect name must not contain whitespace")

    params: dict[str, str] = {}
    for key, value in parse_qsl(parts.query, keep_blank_values=True):
        if not key:
            raise SfxUriError("query parameter names must not be empty")
        if key in params:
            raise SfxUriError(f"duplicate query parameter: {key}")
        params[key] = value
    return SfxSpec(effect=effect, parameters=params)

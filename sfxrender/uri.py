"""Parser for the public ``sfx:`` URI contract."""

from __future__ import annotations

import re
from urllib.parse import parse_qsl, urlsplit

from .errors import SFXURIError, SfxUriError
from .types import SfxSpec

_INVALID_PERCENT_ESCAPE = re.compile(r"%(?![0-9A-Fa-f]{2})")


def parse_sfx_uri(uri: str) -> SfxSpec:
    """Parse ``sfx:effect.name?key=value`` into an :class:`SfxSpec`.

    Raises:
        SFXURIError: If the scheme, effect name, or query syntax is invalid.
    """
    try:
        parts = urlsplit(uri)
    except (TypeError, ValueError) as exc:
        raise SFXURIError(f"Malformed SFX URI: {exc}") from exc

    if parts.scheme.lower() != "sfx":
        raise SFXURIError("SFX URI must use the 'sfx' scheme")
    if parts.netloc:
        raise SFXURIError("SFX URI must not contain an authority component")
    effect = parts.path.strip()
    if not effect:
        raise SFXURIError("SFX URI must contain an effect name")
    if any(ch.isspace() for ch in effect):
        raise SFXURIError("effect name must not contain whitespace")
    if parts.fragment:
        raise SFXURIError("SFX URI must not contain a fragment")
    if _INVALID_PERCENT_ESCAPE.search(parts.path) or _INVALID_PERCENT_ESCAPE.search(parts.query):
        raise SFXURIError("SFX URI contains a malformed percent escape")

    params: dict[str, str] = {}
    try:
        query_params = parse_qsl(
            parts.query,
            keep_blank_values=True,
            strict_parsing=True,
            encoding="utf-8",
            errors="strict",
        )
    except (UnicodeDecodeError, ValueError) as exc:
        raise SFXURIError(f"Malformed SFX URI query: {exc}") from exc
    for key, value in query_params:
        if not key:
            raise SFXURIError("query parameter names must not be empty")
        if key in params:
            raise SFXURIError(f"duplicate query parameter: {key}")
        params[key] = value
    return SfxSpec(effect=effect, parameters=params)


__all__ = ["SFXURIError", "SfxUriError", "parse_sfx_uri"]

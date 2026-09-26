"""Deterministic sound-effect rendering."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from .catalog import catalog
from .core import EffectRenderer, SFXRenderer
from .types import RenderContext, RenderedSound, SfxSpec
from .uri import SfxUriError, parse_sfx_uri

try:
    __version__ = version("sfxrender")
except PackageNotFoundError:  # Source checkout without an installed distribution.
    __version__ = "0+unknown"

__all__ = [
    "EffectRenderer",
    "RenderContext",
    "RenderedSound",
    "SFXRenderer",
    "SfxSpec",
    "SfxUriError",
    "__version__",
    "catalog",
    "parse_sfx_uri",
]

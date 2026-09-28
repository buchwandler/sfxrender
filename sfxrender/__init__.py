"""Deterministic sound-effect rendering."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from .catalog import catalog, llm_catalog
from .core import EffectRenderer, SFXRenderer
from .errors import (
    InvalidEffectParameterError,
    SFXRenderError,
    SFXURIError,
    SfxUriError,
    UnknownEffectError,
)
from .types import RenderContext, RenderedSound, SfxSpec
from .uri import parse_sfx_uri

try:
    __version__ = version("sfxrender")
except PackageNotFoundError:  # Source checkout without an installed distribution.
    __version__ = "0+unknown"

__all__ = [
    "EffectRenderer",
    "InvalidEffectParameterError",
    "RenderContext",
    "RenderedSound",
    "SFXRenderError",
    "SFXRenderer",
    "SFXURIError",
    "SfxSpec",
    "SfxUriError",
    "UnknownEffectError",
    "__version__",
    "catalog",
    "llm_catalog",
    "parse_sfx_uri",
]

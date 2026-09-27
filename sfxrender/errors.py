"""Public exception hierarchy for SFXRender APIs."""

from __future__ import annotations


class SFXRenderError(Exception):
    """Base class for predictable public SFXRender failures."""


class SFXURIError(SFXRenderError, ValueError):
    """Raised when an input does not satisfy the public SFX URI contract."""


# Backwards-compatible spelling retained from the original URI parser API.
SfxUriError = SFXURIError


class UnknownEffectError(SFXRenderError):
    """Raised when a URI or spec names an unregistered effect."""


class InvalidEffectParameterError(SFXRenderError, ValueError):
    """Raised when an effect parameter is unknown or has an invalid value."""

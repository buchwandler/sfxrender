"""Reusable aperiodic contact excitation."""

from __future__ import annotations

import math

import numpy as np

from .._dsp import impact_excitation
from ..types import FloatAudio
from .models import ContactProfile


def contact_excitation(
    *,
    force_envelope: FloatAudio,
    contact: ContactProfile,
    sample_rate: int,
    rng: np.random.Generator,
) -> FloatAudio:
    """Shape broadband impact excitation with perceptual contact properties."""
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    for value, name in (
        (contact.hardness, "hardness"),
        (contact.brightness, "brightness"),
        (contact.roughness, "roughness"),
    ):
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be finite and between 0 and 1")
    if not math.isfinite(contact.contact_gain) or contact.contact_gain < 0.0:
        raise ValueError("contact_gain must be finite and non-negative")

    excitation = impact_excitation(
        sample_rate=sample_rate,
        duration=force_envelope.size,
        force_envelope=force_envelope,
        hardness=contact.hardness,
        brightness=contact.brightness,
        rng=rng,
    )
    # Roughness changes the contact's fine-grained amplitude texture without
    # introducing a free-running tone or a second broadband bed.
    texture_rng = np.random.default_rng(int(rng.integers(0, 2**32, dtype=np.uint32)))
    texture = texture_rng.normal(0.0, 1.0, force_envelope.size).astype(np.float32)
    texture *= np.maximum(force_envelope, 0.0) * np.float32(0.08 * contact.roughness)
    result = (excitation + texture) * np.float32(contact.contact_gain)
    return np.asarray(result, dtype=np.float32)

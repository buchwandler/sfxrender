"""Generic force-shaped impact interactions."""

from __future__ import annotations

import math

import numpy as np

from .._dsp import asymmetric_pulse
from ..types import FloatAudio
from .contacts import contact_excitation
from .models import ContactProfile, ModalBody
from .resonators import resonate


def impact(
    *,
    contact: ContactProfile,
    force: float,
    duration_s: float,
    body: ModalBody | tuple[ModalBody, ...] | None,
    sample_rate: int,
    rng: np.random.Generator,
) -> FloatAudio:
    """Create a contact event and optionally excite an object's modal response."""
    if not math.isfinite(force) or not 0.0 <= force <= 1.0:
        raise ValueError("force must be finite and between 0 and 1")
    if not math.isfinite(duration_s) or duration_s <= 0.0:
        raise ValueError("duration_s must be finite and positive")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")

    size = max(1, round(duration_s * sample_rate))
    attack_s = min(0.003, max(1.0 / sample_rate, duration_s * 0.04))
    decay_s = min(duration_s, max(0.004, duration_s * 0.34))
    envelope = asymmetric_pulse(
        size,
        sample_rate,
        attack_s=attack_s,
        decay_s=decay_s,
        amplitude=force**1.35,
    )
    force_brightness = float(np.clip(contact.brightness + 0.12 * (force - 0.5), 0.0, 1.0))
    force_hardness = float(np.clip(contact.hardness + 0.18 * (force - 0.5), 0.0, 1.0))
    adjusted_contact = ContactProfile(
        hardness=force_hardness,
        brightness=force_brightness,
        roughness=contact.roughness,
        contact_gain=contact.contact_gain,
    )
    result = contact_excitation(
        force_envelope=envelope,
        contact=adjusted_contact,
        sample_rate=sample_rate,
        rng=rng,
    )
    if body is not None:
        bodies = (body,) if isinstance(body, ModalBody) else body
        response_seed = int(rng.integers(0, 2**32, dtype=np.uint32))
        modal_level = 0.45 + 0.35 * force
        for index, resonant_body in enumerate(bodies):
            resonance_rng = np.random.default_rng(np.random.SeedSequence([response_seed, index]))
            body_response = resonate(
                result,
                body=resonant_body,
                sample_rate=sample_rate,
                rng=resonance_rng,
                frequency_jitter=0.012,
                decay_jitter=0.05,
            )
            result = result + body_response * np.float32(modal_level)
    return np.asarray(result, dtype=np.float32)

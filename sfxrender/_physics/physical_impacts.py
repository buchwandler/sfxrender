"""Synthesis helpers for physical contact traces exciting acoustic objects."""

from __future__ import annotations

import math

import numpy as np

from .._dsp import bandpass_noise
from ..types import FloatAudio
from .contact import ContactTrace
from .modes import ModeSet
from .resonator import modal_response


def render_physical_impact(
    trace: ContactTrace,
    modes: ModeSet,
    *,
    sample_rate: int,
    rng: np.random.Generator,
    microscopic_gain: float = 0.5,
    output_gain: float = 24.0,
) -> FloatAudio:
    """Render a contact-force trace through modes plus force-driven microtexture."""
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if not math.isfinite(microscopic_gain) or microscopic_gain < 0.0:
        raise ValueError("microscopic_gain must be finite and non-negative")
    if not math.isfinite(output_gain) or output_gain < 0.0:
        raise ValueError("output_gain must be finite and non-negative")
    force = np.asarray(trace.force_n, dtype=np.float32)
    if force.ndim != 1 or not np.all(np.isfinite(force)) or np.any(force < 0.0):
        raise ValueError("contact force must be a finite non-negative mono signal")
    if force.size == 0 or not modes.modes:
        return np.zeros(force.size, dtype=np.float32)

    peak_force = float(np.max(force))
    if peak_force <= 0.0:
        return np.zeros(force.size, dtype=np.float32)
    high = min(sample_rate * 0.46, 9_000.0)
    low = min(max(120.0, high * 0.18), max(1.0, high - 1.0))
    texture = bandpass_noise(rng, force.size, sample_rate, low, high)
    normalized_force = force / np.float32(peak_force)
    micro_force = texture * normalized_force * np.float32(peak_force * 0.08 * microscopic_gain)
    excitation = np.asarray(force + micro_force, dtype=np.float32)
    tail_s = min(1.25, max(mode.decay_s for mode in modes.modes) * 4.5)
    response = modal_response(excitation, modes, sample_rate, tail_s=tail_s)
    modal_output = response * np.float32(output_gain)
    micro_level = max(float(np.max(np.abs(modal_output))) * 11.0 * microscopic_gain, 1e-5)
    micro_audio = texture * normalized_force * np.float32(micro_level)
    modal_output[: force.size] += micro_audio
    return np.asarray(modal_output, dtype=np.float32)

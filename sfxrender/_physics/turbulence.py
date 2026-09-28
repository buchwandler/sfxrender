"""Seeded band-limited turbulence layers with slow amplitude variation."""

from __future__ import annotations

import math

import numpy as np

from .._dsp import bandpass_noise
from ..types import FloatAudio
from .rng import component_rng

_TURBULENCE_NAMESPACE = 0x54555242


def render_turbulence_layer(
    *,
    size: int,
    sample_rate: int,
    seed: int,
    component_id: int,
    low_hz: float,
    high_hz: float,
    amplitude: float,
    event_index: int = 0,
    variation_depth: float = 0.12,
    envelope: FloatAudio | None = None,
) -> FloatAudio:
    """Render filtered noise with a seeded, slowly varying turbulence envelope."""
    if size < 0:
        raise ValueError("size must be non-negative")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if not math.isfinite(amplitude) or amplitude < 0.0:
        raise ValueError("amplitude must be finite and non-negative")
    if not math.isfinite(variation_depth) or not 0.0 <= variation_depth <= 1.0:
        raise ValueError("variation_depth must be between zero and one")
    if envelope is not None and envelope.size != size:
        raise ValueError("envelope must have the same size as the output")
    if size == 0:
        return np.zeros(0, dtype=np.float32)

    rng = component_rng(seed, _TURBULENCE_NAMESPACE, component_id, event_index)
    noise = bandpass_noise(rng, size, sample_rate, low_hz, high_hz)
    spacing = max(1, round(0.45 * sample_rate))
    knot_positions = np.arange(0, size, spacing, dtype=np.float64)
    if knot_positions[-1] != size - 1:
        knot_positions = np.append(knot_positions, size - 1)
    knots = rng.uniform(-1.0, 1.0, knot_positions.size).astype(np.float32)
    control = np.interp(np.arange(size, dtype=np.float64), knot_positions, knots).astype(np.float32)
    modulation = np.float32(1.0) + control * np.float32(variation_depth)
    if envelope is not None:
        modulation *= envelope
    return np.asarray(noise * modulation * np.float32(amplitude), dtype=np.float32)

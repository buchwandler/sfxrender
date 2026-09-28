"""Low-level electrical and electromechanical hum sources."""

from __future__ import annotations

import math

import numpy as np

from ..types import FloatAudio
from .rng import component_rng

_HUM_NAMESPACE = 0x454C4543


def render_electromechanical_hum(
    *,
    sample_rate: int,
    size: int,
    seed: int,
    event_index: int = 0,
    fundamental_hz: float = 60.0,
    amplitude: float = 0.02,
    envelope: FloatAudio | None = None,
    harmonic_gains: tuple[float, ...] = (1.0, 0.28, 0.12, 0.06),
) -> FloatAudio:
    """Render restrained mains-like hum with seeded harmonic phase variation."""
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if size < 0:
        raise ValueError("size must be non-negative")
    if not math.isfinite(fundamental_hz) or fundamental_hz <= 0.0:
        raise ValueError("fundamental_hz must be finite and positive")
    if not math.isfinite(amplitude) or amplitude < 0.0:
        raise ValueError("amplitude must be finite and non-negative")
    if envelope is not None and (envelope.ndim != 1 or envelope.size != size):
        raise ValueError("envelope must match the requested signal size")
    if not harmonic_gains or any(not math.isfinite(gain) or gain < 0.0 for gain in harmonic_gains):
        raise ValueError("harmonic_gains must contain finite non-negative values")
    if size == 0:
        return np.zeros(0, dtype=np.float32)

    rng = component_rng(seed, _HUM_NAMESPACE, 1, event_index)
    time = np.arange(size, dtype=np.float64) / sample_rate
    signal = np.zeros(size, dtype=np.float64)
    for harmonic, gain in enumerate(harmonic_gains, start=1):
        frequency = fundamental_hz * harmonic
        if frequency >= sample_rate * 0.46:
            break
        phase = float(rng.uniform(-math.pi, math.pi))
        signal += gain * np.sin(2.0 * math.pi * frequency * time + phase)

    if envelope is not None:
        signal *= np.clip(envelope, 0.0, 1.0)
    return np.asarray(signal * amplitude, dtype=np.float32)

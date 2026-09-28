"""Speed-driven harmonic sources for rotating mechanical assemblies."""

from __future__ import annotations

import math

import numpy as np

from ..types import FloatAudio
from .rng import component_rng

_ROTATING_NAMESPACE = 0x524F544D


def render_rotating_machine(
    *,
    speed_curve_hz: FloatAudio,
    sample_rate: int,
    seed: int,
    event_index: int = 0,
    ripple_rate_hz: float = 12.0,
    ripple_depth: float = 0.16,
    amplitude: float = 0.2,
    harmonic_gains: tuple[float, ...] = (1.0, 0.42, 0.23, 0.12),
) -> FloatAudio:
    """Render harmonics whose instantaneous pitch follows a mechanical speed curve."""
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if speed_curve_hz.ndim != 1 or not np.isfinite(speed_curve_hz).all():
        raise ValueError("speed_curve_hz must be a finite one-dimensional array")
    if np.any(speed_curve_hz < 0.0):
        raise ValueError("speed_curve_hz must be non-negative")
    if not math.isfinite(ripple_rate_hz) or ripple_rate_hz <= 0.0:
        raise ValueError("ripple_rate_hz must be finite and positive")
    if not math.isfinite(ripple_depth) or not 0.0 <= ripple_depth < 1.0:
        raise ValueError("ripple_depth must be finite and in [0, 1)")
    if not math.isfinite(amplitude) or amplitude < 0.0:
        raise ValueError("amplitude must be finite and non-negative")
    if not harmonic_gains or any(not math.isfinite(gain) or gain < 0.0 for gain in harmonic_gains):
        raise ValueError("harmonic_gains must contain finite non-negative values")

    size = speed_curve_hz.size
    if size == 0:
        return np.zeros(0, dtype=np.float32)
    speed = speed_curve_hz.astype(np.float64)
    phase = np.zeros(size, dtype=np.float64)
    if size > 1:
        phase[1:] = np.cumsum(2.0 * math.pi * speed[:-1] / sample_rate)

    rng = component_rng(seed, _ROTATING_NAMESPACE, 1, event_index)
    tone = np.zeros(size, dtype=np.float64)
    nyquist_limit = sample_rate * 0.46
    for harmonic, gain in enumerate(harmonic_gains, start=1):
        active = speed * harmonic < nyquist_limit
        if not np.any(active):
            continue
        offset = float(rng.uniform(-math.pi, math.pi))
        tone += np.sin(phase * harmonic + offset) * gain * active

    time = np.arange(size, dtype=np.float64) / sample_rate
    ripple_phase = 2.0 * math.pi * ripple_rate_hz * time + float(rng.uniform(-math.pi, math.pi))
    ripple = 1.0 - ripple_depth + ripple_depth * np.sin(ripple_phase)
    return np.asarray(tone * ripple * amplitude, dtype=np.float32)

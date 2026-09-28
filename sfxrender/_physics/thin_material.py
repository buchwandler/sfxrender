"""Seeded flutter and crinkle sources for thin resonant sheets."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .._dsp import bandpass_noise, mix_at
from ..types import FloatAudio
from .modes import Mode, ModeSet
from .resonator import modal_response
from .rng import component_rng

_THIN_MATERIAL_NAMESPACE = 0x54484E4D


@dataclass(frozen=True, slots=True)
class ThinMaterialPreset:
    """A thin sheet's broad flutter band and lightly damped mode family."""

    modes: ModeSet
    noise_band_hz: tuple[float, float]
    event_rate_hz: float
    tail_s: float = 0.16


def generate_thin_material_preset(*, seed: int, sample_rate: int) -> ThinMaterialPreset:
    """Generate a stable paper-like sheet response for a seed and output rate."""
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    rng = component_rng(seed, _THIN_MATERIAL_NAMESPACE, 1, sample_rate)
    specifications = (
        (360.0, 0.055, 0.44),
        (720.0, 0.042, 0.35),
        (1_240.0, 0.032, 0.25),
        (1_980.0, 0.024, 0.17),
        (2_850.0, 0.018, 0.10),
    )
    modes = ModeSet(
        tuple(
            Mode(
                frequency * float(rng.uniform(0.98, 1.02)),
                decay * float(rng.uniform(0.9, 1.1)),
                input_gain=float(gain * rng.uniform(0.88, 1.12)),
                radiation_gain=gain,
                modal_mass_kg=0.025 + index * 0.008,
            )
            for index, (frequency, decay, gain) in enumerate(specifications)
            if frequency < sample_rate * 0.43
        )
    )
    return ThinMaterialPreset(
        modes=modes,
        noise_band_hz=(350.0, min(7_200.0, sample_rate * 0.44)),
        event_rate_hz=float(rng.uniform(16.0, 32.0)),
    )


def render_thin_material_source(
    *,
    preset: ThinMaterialPreset,
    activity: FloatAudio,
    sample_rate: int,
    seed: int,
    event_index: int = 0,
    amplitude: float = 0.18,
) -> FloatAudio:
    """Render broadband sheet friction, irregular flutter bursts, and body modes."""
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if activity.ndim != 1 or not np.isfinite(activity).all():
        raise ValueError("activity must be a finite one-dimensional array")
    if not math.isfinite(amplitude) or amplitude < 0.0:
        raise ValueError("amplitude must be finite and non-negative")
    size = activity.size
    if size == 0:
        return np.zeros(0, dtype=np.float32)
    envelope = np.clip(activity, 0.0, 1.0).astype(np.float32)
    if not np.any(envelope):
        return np.zeros(size, dtype=np.float32)

    rng = component_rng(seed, _THIN_MATERIAL_NAMESPACE, 2, event_index)
    low, high = preset.noise_band_hz
    safe_high = min(high, sample_rate * 0.44)
    bed = bandpass_noise(rng, size, sample_rate, low, safe_high)
    result = np.asarray(bed * envelope, dtype=np.float32)
    mean_activity = float(np.mean(envelope))
    event_count = round(size / sample_rate * preset.event_rate_hz * mean_activity)
    weights = envelope.astype(np.float64) + 1e-4
    weights /= weights.sum()
    burst_low = min(1_200.0, safe_high * 0.4)
    for _ in range(event_count):
        start = int(rng.choice(size, p=weights))
        burst_size = min(
            size - start,
            max(2, round(float(rng.uniform(0.003, 0.015)) * sample_rate)),
        )
        if burst_size <= 0:
            continue
        burst = bandpass_noise(
            rng,
            burst_size,
            sample_rate,
            max(80.0, min(burst_low, safe_high * 0.3)),
            safe_high,
        )
        time = np.arange(burst_size, dtype=np.float32) / np.float32(sample_rate)
        decay_s = float(rng.uniform(0.004, 0.018))
        burst *= np.exp(-time / np.float32(decay_s))
        mix_at(result, burst * np.float32(0.22 * envelope[start]), start)

    resonant = modal_response(result, preset.modes, sample_rate, tail_s=preset.tail_s)
    output = np.zeros(resonant.size, dtype=np.float32)
    mix_at(output, result, 0)
    mix_at(output, resonant * np.float32(0.24), 0)
    return np.asarray(output * np.float32(amplitude), dtype=np.float32)

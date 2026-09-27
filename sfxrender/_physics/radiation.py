"""Lightweight mode-dependent acoustic radiation weighting."""

from __future__ import annotations

import math


def radiation_efficiency(
    frequency_hz: float,
    characteristic_size_m: float,
    *,
    speed_of_sound_m_s: float = 343.0,
) -> float:
    """Smoothly suppress radiation from small bodies at low frequencies."""
    if not math.isfinite(frequency_hz) or frequency_hz <= 0.0:
        raise ValueError("frequency_hz must be finite and positive")
    if not math.isfinite(characteristic_size_m) or characteristic_size_m <= 0.0:
        raise ValueError("characteristic_size_m must be finite and positive")
    if not math.isfinite(speed_of_sound_m_s) or speed_of_sound_m_s <= 0.0:
        raise ValueError("speed_of_sound_m_s must be finite and positive")
    ka = 2.0 * math.pi * frequency_hz * characteristic_size_m / speed_of_sound_m_s
    return ka / math.sqrt(1.0 + ka * ka)

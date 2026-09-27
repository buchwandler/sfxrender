"""Seeded modal-body generation and excitation-driven object resonance."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from .._dsp import modal_bank
from ..types import FloatAudio
from .models import ModalBody, Mode, ModeBand


def generate_modal_body(
    *,
    base_bands: Sequence[ModeBand],
    size_scale: float,
    damping_scale: float,
    brightness_scale: float,
    rng: np.random.Generator,
    sample_rate: int,
) -> ModalBody:
    """Sample a plausible correlated mode family from semantic ranges.

    ``size_scale`` shifts every mode together (larger objects trend lower); it is
    not a dimensional mechanics solver. Modes too close to or above Nyquist are
    omitted, and a modest spacing guard avoids accidental single-note clusters.
    """
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    for value, name in (
        (size_scale, "size_scale"),
        (damping_scale, "damping_scale"),
        (brightness_scale, "brightness_scale"),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")

    size_shift = math.exp(float(rng.normal(0.0, 0.035))) / size_scale
    nyquist_limit = sample_rate * 0.45
    sampled: list[Mode] = []
    occupied: list[float] = []
    for band in base_bands:
        low = band.low_hz * size_shift
        high = min(band.high_hz * size_shift, nyquist_limit)
        if high <= low:
            continue
        count = int(rng.integers(band.count[0], band.count[1] + 1))
        candidates: list[float] = []
        for _ in range(count):
            candidate = 0.0
            for _attempt in range(10):
                candidate = float(np.exp(rng.uniform(math.log(low), math.log(high))))
                spacing = max(2.0, candidate * 0.015)
                if all(abs(candidate - prior) >= spacing for prior in occupied):
                    break
            else:
                continue
            occupied.append(candidate)
            candidates.append(candidate)

        for frequency in candidates:
            gain = float(rng.uniform(band.gain[0], band.gain[1]))
            normalized_height = min(
                1.0, max(0.0, math.log(frequency / 40.0) / math.log(max(2.0, nyquist_limit / 40.0)))
            )
            gain *= brightness_scale ** (0.45 + normalized_height)
            decay = float(rng.uniform(band.decay_s[0], band.decay_s[1]))
            sampled.append(Mode(frequency_hz=frequency, decay_s=decay, gain=gain))

    sampled.sort(key=lambda mode: mode.frequency_hz)
    radiation_gain = float(rng.uniform(0.88, 1.12))
    return ModalBody(
        modes=tuple(sampled),
        radiation_gain=radiation_gain,
        damping_scale=damping_scale,
    )


def resonate(
    excitation: FloatAudio,
    *,
    body: ModalBody,
    sample_rate: int,
    rng: np.random.Generator,
    frequency_jitter: float = 0.0,
    decay_jitter: float = 0.0,
) -> FloatAudio:
    """Drive a generated resonant body with an excitation using ``modal_bank``."""
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if (
        not math.isfinite(frequency_jitter)
        or frequency_jitter < 0.0
        or not math.isfinite(decay_jitter)
        or decay_jitter < 0.0
    ):
        raise ValueError("jitter values must be finite and non-negative")
    if excitation.size == 0 or not body.modes:
        return np.zeros(excitation.size, dtype=np.float32)

    # Leave a safety margin for the modal_bank's per-render frequency jitter.
    safe_ceiling = sample_rate * 0.48 / (1.0 + frequency_jitter)
    modes = tuple(
        (mode.frequency_hz, mode.decay_s * body.damping_scale, mode.gain)
        for mode in body.modes
        if mode.frequency_hz < safe_ceiling
    )
    if not modes:
        return np.zeros(excitation.size, dtype=np.float32)
    result = modal_bank(
        excitation,
        sample_rate,
        rng,
        modes,
        frequency_jitter=frequency_jitter,
        decay_jitter=decay_jitter,
    )
    return np.asarray(result * np.float32(body.radiation_gain), dtype=np.float32)

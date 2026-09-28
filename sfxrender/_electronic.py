"""Reusable deterministic electronic tone synthesis."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ._dsp import mix_at, one_pole_highpass, one_pole_lowpass
from .types import FloatAudio


@dataclass(frozen=True, slots=True)
class TonePreset:
    """A compact tonal source preset independent of any semantic effect."""

    frequency_hz: float
    harmonic_gains: tuple[float, ...]
    attack_s: float
    release_s: float
    bandwidth_hz: tuple[float, float] | None = None
    distortion: float = 0.0
    transient_click: float = 0.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.frequency_hz) or self.frequency_hz <= 0.0:
            raise ValueError("frequency_hz must be finite and positive")
        if not self.harmonic_gains or any(
            not math.isfinite(gain) or gain < 0.0 for gain in self.harmonic_gains
        ):
            raise ValueError("harmonic_gains must contain finite non-negative values")
        if not math.isfinite(self.attack_s) or self.attack_s < 0.0:
            raise ValueError("attack_s must be finite and non-negative")
        if not math.isfinite(self.release_s) or self.release_s < 0.0:
            raise ValueError("release_s must be finite and non-negative")
        if self.bandwidth_hz is not None:
            low, high = self.bandwidth_hz
            if not math.isfinite(low) or not math.isfinite(high) or low <= 0.0 or high <= low:
                raise ValueError("bandwidth_hz must be a finite increasing positive pair")
        if not math.isfinite(self.distortion) or not 0.0 <= self.distortion <= 1.0:
            raise ValueError("distortion must be finite and between zero and one")
        if not math.isfinite(self.transient_click) or self.transient_click < 0.0:
            raise ValueError("transient_click must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class TonePulse:
    """One pulse of a reusable tone source."""

    start_s: float
    duration_s: float
    level: float = 1.0
    pitch_scale: float = 1.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.start_s) or self.start_s < 0.0:
            raise ValueError("start_s must be finite and non-negative")
        if not math.isfinite(self.duration_s) or self.duration_s <= 0.0:
            raise ValueError("duration_s must be finite and positive")
        if not math.isfinite(self.level) or self.level < 0.0:
            raise ValueError("level must be finite and non-negative")
        if not math.isfinite(self.pitch_scale) or self.pitch_scale <= 0.0:
            raise ValueError("pitch_scale must be finite and positive")


def render_tone_pattern(
    *,
    preset: TonePreset,
    pulses: Sequence[TonePulse],
    duration_s: float,
    sample_rate: int,
    rng: np.random.Generator,
) -> FloatAudio:
    """Render seeded harmonic tone pulses with optional transducer coloration."""
    if not math.isfinite(duration_s) or duration_s <= 0.0:
        raise ValueError("duration_s must be finite and positive")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a NumPy Generator")
    size = max(1, round(duration_s * sample_rate))
    output = np.zeros(size, dtype=np.float32)

    for pulse in pulses:
        if not isinstance(pulse, TonePulse):
            raise TypeError("pulses must contain TonePulse values")
        pulse_size = min(round(pulse.duration_s * sample_rate), size - round(pulse.start_s * sample_rate))
        if pulse_size <= 0:
            continue
        start = round(pulse.start_s * sample_rate)
        time = np.arange(pulse_size, dtype=np.float64) / sample_rate
        phase = float(rng.uniform(-math.pi, math.pi))
        tone = np.zeros(pulse_size, dtype=np.float64)
        for harmonic, gain in enumerate(preset.harmonic_gains, start=1):
            frequency = preset.frequency_hz * pulse.pitch_scale * harmonic
            if frequency >= sample_rate * 0.49:
                continue
            tone += gain * np.sin(2.0 * math.pi * frequency * time + phase * harmonic)
        tone = np.asarray(tone, dtype=np.float32)
        if preset.distortion > 0.0:
            tone = np.asarray(np.tanh(tone * (1.0 + 5.0 * preset.distortion)), dtype=np.float32)
        if preset.bandwidth_hz is not None:
            low, high = preset.bandwidth_hz
            tone = one_pole_highpass(tone, sample_rate, low)
            tone = one_pole_lowpass(tone, sample_rate, high)

        envelope = np.ones(pulse_size, dtype=np.float32)
        attack = min(pulse_size, round(preset.attack_s * sample_rate))
        release = min(pulse_size, round(preset.release_s * sample_rate))
        if attack:
            envelope[:attack] *= np.linspace(0.0, 1.0, attack, dtype=np.float32)
        if release:
            envelope[-release:] *= np.linspace(1.0, 0.0, release, dtype=np.float32)
        tone *= envelope * np.float32(pulse.level)
        mix_at(output, tone, start)
        if preset.transient_click > 0.0 and start < size:
            output[start] += np.float32(preset.transient_click * pulse.level)

    return output

"""Small immutable value types for perceptual physical sound models."""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..types import FloatAudio


def _finite_positive(value: float, name: str) -> None:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")


def _finite_nonnegative(value: float, name: str) -> None:
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class Mode:
    """One damped resonant mode."""

    frequency_hz: float
    decay_s: float
    gain: float

    def __post_init__(self) -> None:
        _finite_positive(self.frequency_hz, "frequency_hz")
        _finite_positive(self.decay_s, "decay_s")
        _finite_nonnegative(self.gain, "gain")


@dataclass(frozen=True, slots=True)
class ModeBand:
    """Ranges from which a seeded generator draws a small mode family."""

    low_hz: float
    high_hz: float
    count: tuple[int, int]
    gain: tuple[float, float]
    decay_s: tuple[float, float]

    def __post_init__(self) -> None:
        _finite_positive(self.low_hz, "low_hz")
        _finite_positive(self.high_hz, "high_hz")
        if self.high_hz <= self.low_hz:
            raise ValueError("high_hz must be greater than low_hz")
        if len(self.count) != 2 or self.count[0] < 0 or self.count[1] < self.count[0]:
            raise ValueError("count must be an ordered pair of non-negative integers")
        if len(self.gain) != 2:
            raise ValueError("gain must contain a minimum and maximum")
        _finite_nonnegative(self.gain[0], "gain minimum")
        _finite_nonnegative(self.gain[1], "gain maximum")
        if self.gain[1] < self.gain[0]:
            raise ValueError("gain maximum must not be less than its minimum")
        if len(self.decay_s) != 2:
            raise ValueError("decay_s must contain a minimum and maximum")
        _finite_positive(self.decay_s[0], "decay_s minimum")
        _finite_positive(self.decay_s[1], "decay_s maximum")
        if self.decay_s[1] < self.decay_s[0]:
            raise ValueError("decay_s maximum must not be less than its minimum")


@dataclass(frozen=True, slots=True)
class ModalBody:
    """A generated resonant object and its broad radiation/damping controls."""

    modes: tuple[Mode, ...]
    radiation_gain: float = 1.0
    damping_scale: float = 1.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.radiation_gain) or self.radiation_gain < 0.0:
            raise ValueError("radiation_gain must be finite and non-negative")
        _finite_positive(self.damping_scale, "damping_scale")


@dataclass(frozen=True, slots=True)
class ContactProfile:
    """Perceptual hardness and spectral character of a contact."""

    hardness: float
    brightness: float
    roughness: float
    contact_gain: float


@dataclass(frozen=True, slots=True)
class FrictionProfile:
    """Perceptual controls for sliding or rotational friction."""

    base_gain: float
    roughness: float
    noise_band_hz: tuple[float, float]
    stick_strength: float
    slip_strength: float
    f0_hz: tuple[float, float]
    harmonic_rolloff: tuple[float, float]
    chaos_amount: float


@dataclass(frozen=True, slots=True)
class MotionCurve:
    """Normalized position and its time derivatives for one action."""

    position: FloatAudio
    velocity: FloatAudio
    acceleration: FloatAudio

    def __post_init__(self) -> None:
        if self.position.ndim != 1 or self.velocity.ndim != 1 or self.acceleration.ndim != 1:
            raise ValueError("motion arrays must be one-dimensional")
        if not (self.position.size == self.velocity.size == self.acceleration.size):
            raise ValueError("motion arrays must have equal lengths")

"""Modal value types, filtering and explicit damping conversions."""

from __future__ import annotations

import math
from dataclasses import dataclass


def q_from_decay(frequency_hz: float, decay_s: float) -> float:
    """Convert amplitude e-folding time to the lightly-damped oscillator Q."""
    if not math.isfinite(frequency_hz) or frequency_hz <= 0.0:
        raise ValueError("frequency_hz must be finite and positive")
    if not math.isfinite(decay_s) or decay_s <= 0.0:
        raise ValueError("decay_s must be finite and positive")
    return math.pi * frequency_hz * decay_s


def decay_from_q(frequency_hz: float, q: float) -> float:
    """Convert lightly-damped oscillator Q to amplitude e-folding time."""
    if not math.isfinite(frequency_hz) or frequency_hz <= 0.0:
        raise ValueError("frequency_hz must be finite and positive")
    if not math.isfinite(q) or q <= 0.0:
        raise ValueError("q must be finite and positive")
    return q / (math.pi * frequency_hz)


@dataclass(frozen=True, slots=True)
class Mode:
    """A damped object mode with force participation and acoustic radiation."""

    frequency_hz: float
    decay_s: float
    input_gain: float = 1.0
    radiation_gain: float = 1.0
    modal_mass_kg: float = 1.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.frequency_hz) or self.frequency_hz <= 0.0:
            raise ValueError("frequency_hz must be finite and positive")
        if not math.isfinite(self.decay_s) or self.decay_s <= 0.0:
            raise ValueError("decay_s must be finite and positive")
        if not math.isfinite(self.input_gain):
            raise ValueError("input_gain must be finite")
        if not math.isfinite(self.radiation_gain) or self.radiation_gain < 0.0:
            raise ValueError("radiation_gain must be finite and non-negative")
        if not math.isfinite(self.modal_mass_kg) or self.modal_mass_kg <= 0.0:
            raise ValueError("modal_mass_kg must be finite and positive")


@dataclass(frozen=True, slots=True)
class ModeSet:
    """Immutable ordered collection of object modes."""

    modes: tuple[Mode, ...]

    def __post_init__(self) -> None:
        if not all(isinstance(mode, Mode) for mode in self.modes):
            raise TypeError("modes must contain only Mode values")


def filter_audio_modes(
    modes: ModeSet,
    *,
    sample_rate: int,
    nyquist_fraction: float = 0.46,
) -> ModeSet:
    """Return modes safely below the selected Nyquist fraction."""
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if not math.isfinite(nyquist_fraction) or not 0.0 < nyquist_fraction < 0.5:
        raise ValueError("nyquist_fraction must be finite and between zero and 0.5")
    limit = sample_rate * nyquist_fraction
    return ModeSet(tuple(mode for mode in modes.modes if mode.frequency_hz < limit))

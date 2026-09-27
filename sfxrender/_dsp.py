"""Small NumPy-only DSP building blocks shared by procedural Foley effects."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from .types import FloatAudio


def unit_rms(signal: FloatAudio) -> FloatAudio:
    """Return a float32 copy scaled to unit RMS, preserving silence."""
    rms = float(np.sqrt(np.mean(np.square(signal, dtype=np.float64)))) if signal.size else 0.0
    if rms <= 1e-12:
        return signal.copy()
    return np.asarray(signal / np.float32(rms), dtype=np.float32)


def one_pole_lowpass(signal: FloatAudio, sample_rate: int, cutoff_hz: float) -> FloatAudio:
    """Apply a stable one-pole low-pass filter."""
    if signal.size == 0:
        return signal.copy()
    cutoff = float(np.clip(cutoff_hz, 1.0, sample_rate * 0.49))
    coefficient = 1.0 - math.exp(-2.0 * math.pi * cutoff / sample_rate)
    output = np.empty(signal.size, dtype=np.float32)
    state = 0.0
    for index, sample in enumerate(signal):
        state += coefficient * (float(sample) - state)
        output[index] = state
    return output


def one_pole_highpass(signal: FloatAudio, sample_rate: int, cutoff_hz: float) -> FloatAudio:
    """Apply a one-pole high-pass by subtracting a matched low-pass."""
    return np.asarray(signal - one_pole_lowpass(signal, sample_rate, cutoff_hz), dtype=np.float32)


def bandpass_noise(
    rng: np.random.Generator,
    size: int,
    sample_rate: int,
    low_hz: float,
    high_hz: float,
) -> FloatAudio:
    """Create deterministic unit-RMS noise between two cutoff frequencies."""
    if size <= 0:
        return np.zeros(0, dtype=np.float32)
    low = max(1.0, min(float(low_hz), sample_rate * 0.48))
    high = max(low + 1.0, min(float(high_hz), sample_rate * 0.49))
    white = rng.normal(0.0, 1.0, size).astype(np.float32)
    shaped = one_pole_lowpass(white, sample_rate, high)
    shaped -= one_pole_lowpass(white, sample_rate, low)
    return unit_rms(np.asarray(shaped, dtype=np.float32))


def asymmetric_pulse(
    size: int,
    sample_rate: int,
    *,
    onset_s: float = 0.0,
    attack_s: float,
    decay_s: float,
    amplitude: float = 1.0,
) -> FloatAudio:
    """Generate a compact asymmetric attack/decay pulse of ``size`` samples."""
    pulse = np.zeros(max(0, int(size)), dtype=np.float32)
    start = max(0, round(onset_s * sample_rate))
    if start >= pulse.size:
        return pulse
    t = np.arange(pulse.size - start, dtype=np.float32) / np.float32(sample_rate)
    attack = max(float(attack_s), 1.0 / sample_rate)
    decay = max(float(decay_s), 1.0 / sample_rate)
    shape = (1.0 - np.exp(-t / np.float32(attack))) * np.exp(-t / np.float32(decay))
    pulse[start:] = shape * np.float32(amplitude)
    return pulse


def envelope_from_points(
    points: Sequence[tuple[float, float]], sample_rate: int, size: int
) -> FloatAudio:
    """Linearly interpolate an amplitude envelope from time/value control points."""
    if size <= 0:
        return np.zeros(0, dtype=np.float32)
    if not points:
        return np.zeros(size, dtype=np.float32)
    times = np.asarray([max(0.0, point[0]) for point in points], dtype=np.float64)
    values = np.asarray([point[1] for point in points], dtype=np.float64)
    order = np.argsort(times, kind="stable")
    sample_times = np.arange(size, dtype=np.float64) / float(sample_rate)
    return np.interp(sample_times, times[order], values[order]).astype(np.float32)


def mix_at(destination: FloatAudio, source: FloatAudio, start: int) -> None:
    """Add a source signal into a destination, clipping at either end."""
    if start < 0:
        source = source[-start:]
        start = 0
    if start >= destination.size or source.size == 0:
        return
    end = min(destination.size, start + source.size)
    destination[start:end] += source[: end - start]


def modal_bank(
    excitation: FloatAudio,
    sample_rate: int,
    rng: np.random.Generator,
    modes: Sequence[tuple[float, float, float]],
    *,
    frequency_jitter: float = 0.02,
    decay_jitter: float = 0.08,
) -> FloatAudio:
    """Resonate contact excitation through seeded, damped inharmonic modes.

    Each mode is ``(frequency_hz, decay_seconds, gain)``. Frequency and decay
    vary slightly per impact, while the input excitation—not a free-running
    oscillator—supplies the resonator energy.
    """
    output = np.zeros(excitation.size, dtype=np.float32)
    if excitation.size == 0 or not modes:
        return output
    fft_size = 1 << max(1, (2 * excitation.size - 1).bit_length())
    excitation_spectrum = np.fft.rfft(excitation, n=fft_size)
    time = np.arange(excitation.size, dtype=np.float32) / np.float32(sample_rate)
    for frequency, decay, gain in modes:
        varied_frequency = float(frequency) * float(
            rng.uniform(1.0 - frequency_jitter, 1.0 + frequency_jitter)
        )
        varied_decay = max(
            1.0 / sample_rate,
            float(decay) * float(rng.uniform(1.0 - decay_jitter, 1.0 + decay_jitter)),
        )
        impulse = np.sin(np.float32(2.0 * math.pi * varied_frequency) * time)
        impulse *= np.exp(-time / np.float32(varied_decay))
        norm = float(np.sqrt(np.sum(np.square(impulse, dtype=np.float64))))
        if norm <= 1e-12:
            continue
        impulse /= np.float32(norm)
        response = np.fft.irfft(excitation_spectrum * np.fft.rfft(impulse, n=fft_size), n=fft_size)[
            : excitation.size
        ]
        output += np.asarray(response * np.float32(gain), dtype=np.float32)
    return output


def impact_excitation(
    *,
    sample_rate: int,
    duration: int,
    force_envelope: FloatAudio,
    hardness: float,
    brightness: float,
    rng: np.random.Generator,
) -> FloatAudio:
    """Create an aperiodic, envelope-shaped broadband contact excitation."""
    size = max(0, int(duration))
    if size == 0:
        return np.zeros(0, dtype=np.float32)
    envelope = np.zeros(size, dtype=np.float32)
    count = min(size, force_envelope.size)
    if count:
        envelope[:count] = np.maximum(force_envelope[:count], 0.0)
    low = bandpass_noise(rng, size, sample_rate, 45.0, 900.0)
    mid = bandpass_noise(rng, size, sample_rate, 350.0, 3_200.0)
    high = bandpass_noise(rng, size, sample_rate, 1_600.0, sample_rate * 0.46)
    hard = float(np.clip(hardness, 0.0, 1.0))
    bright = float(np.clip(brightness, 0.0, 1.0))
    texture = (0.62 - 0.32 * hard) * low + (0.75 - 0.22 * abs(bright - 0.5)) * mid
    texture += (0.10 + 0.55 * hard * bright) * high
    return np.asarray(texture * envelope, dtype=np.float32)

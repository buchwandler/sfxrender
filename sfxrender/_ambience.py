"""Finite room, office, and city ambience compositions."""

from __future__ import annotations

import math

import numpy as np

from ._dsp import bandpass_noise, mix_at
from ._physics.electromechanical import render_electromechanical_hum
from ._physics.rng import component_rng
from ._physics.stochastic import StochasticEvent, sample_stochastic_events
from .types import FloatAudio

_AMBIENCE_NAMESPACE = 0x414D4249
_ROOM_PROFILES = {
    "quiet": (1_800.0, 0.010, 0.006, 0.0025),
    "ventilated": (4_200.0, 0.009, 0.008, 0.0035),
    "electrical": (2_800.0, 0.006, 0.005, 0.010),
}


def _room_tone(*, sample_rate: int, duration: float, profile: str, seed: int) -> FloatAudio:
    size = max(1, round(duration * sample_rate))
    high_hz, bed_gain, low_gain, hum_gain = _ROOM_PROFILES[profile]
    rng = component_rng(seed, _AMBIENCE_NAMESPACE, 1, sample_rate)
    bed = bandpass_noise(rng, size, sample_rate, 120.0, high_hz)
    low = bandpass_noise(rng, size, sample_rate, 35.0, 420.0)
    time = np.arange(size, dtype=np.float32) / np.float32(sample_rate)
    phase = float(rng.uniform(-math.pi, math.pi))
    modulation = 0.94 + 0.035 * np.sin(2.0 * math.pi * 0.11 * time + phase)
    modulation += 0.02 * np.sin(2.0 * math.pi * 0.23 * time + phase * 0.7)
    result = np.asarray((bed * bed_gain + low * low_gain) * modulation, dtype=np.float32)
    hum = render_electromechanical_hum(
        sample_rate=sample_rate,
        size=size,
        seed=seed,
        event_index=10,
        fundamental_hz=60.0,
        amplitude=hum_gain,
        harmonic_gains=(1.0, 0.28, 0.12, 0.06),
    )
    mix_at(result, hum, 0)
    return result


def render_room_tone(*, sample_rate: int, duration: float, character: str, seed: int) -> FloatAudio:
    return _room_tone(sample_rate=sample_rate, duration=duration, profile=character, seed=seed)


def _office_event(
    event: StochasticEvent, *, sample_rate: int, rng: np.random.Generator
) -> FloatAudio:
    kind = int(rng.integers(0, 3))
    size = event.duration_samples
    if kind == 0:
        band = (1_400.0, min(5_800.0, sample_rate * 0.44))
        decay_s = 0.012
        gain = 0.010
    elif kind == 1:
        band = (320.0, min(3_400.0, sample_rate * 0.44))
        decay_s = 0.055
        gain = 0.006
    else:
        band = (75.0, min(950.0, sample_rate * 0.44))
        decay_s = 0.040
        gain = 0.007
    noise = bandpass_noise(rng, size, sample_rate, *band)
    time = np.arange(size, dtype=np.float32) / np.float32(sample_rate)
    envelope = np.exp(-time / np.float32(decay_s))
    return np.asarray(noise * envelope * np.float32(gain * event.level), dtype=np.float32)


def render_office_ambience(
    *, sample_rate: int, duration: float, activity: str, seed: int
) -> FloatAudio:
    size = max(1, round(duration * sample_rate))
    result = _room_tone(sample_rate=sample_rate, duration=duration, profile="ventilated", seed=seed)
    result *= np.float32(0.76)
    rate_hz = {"quiet": 0.55, "busy": 1.65}[activity]
    events = sample_stochastic_events(
        duration_s=duration,
        sample_rate=sample_rate,
        rate_hz=rate_hz,
        seed=seed,
        component_id=20,
        event_duration_s=(0.012, 0.14),
    )
    rng = component_rng(seed, _AMBIENCE_NAMESPACE, 2, sample_rate)
    for event in events:
        burst = _office_event(event, sample_rate=sample_rate, rng=rng)
        mix_at(result, burst, event.start_sample)
    return result[:size]


def _city_passby(
    event: StochasticEvent, *, sample_rate: int, rng: np.random.Generator
) -> FloatAudio:
    size = event.duration_samples
    low = bandpass_noise(rng, size, sample_rate, 45.0, 360.0)
    mid = bandpass_noise(rng, size, sample_rate, 180.0, min(1_600.0, sample_rate * 0.42))
    position = np.linspace(0.0, 1.0, size, dtype=np.float32)
    envelope = np.maximum(np.sin(np.float32(math.pi) * position), np.float32(0.0)) ** np.float32(
        1.4
    )
    level = np.float32(0.014 * event.level)
    return np.asarray(
        (low * np.float32(0.72) + mid * np.float32(0.28)) * envelope * level, dtype=np.float32
    )


def render_city_ambience(
    *, sample_rate: int, duration: float, activity: str, seed: int
) -> FloatAudio:
    size = max(1, round(duration * sample_rate))
    rng = component_rng(seed, _AMBIENCE_NAMESPACE, 3, sample_rate)
    low = bandpass_noise(rng, size, sample_rate, 28.0, 230.0)
    mid = bandpass_noise(rng, size, sample_rate, 120.0, 1_100.0)
    time = np.arange(size, dtype=np.float32) / np.float32(sample_rate)
    phase = float(rng.uniform(-math.pi, math.pi))
    traffic_bed = low * np.float32(0.014) + mid * np.float32(0.006)
    traffic_bed *= 0.87 + 0.09 * np.sin(2.0 * math.pi * 0.035 * time + phase)
    result = np.asarray(traffic_bed, dtype=np.float32)
    rate_hz = {"calm": 0.22, "busy": 0.62}[activity]
    events = sample_stochastic_events(
        duration_s=duration,
        sample_rate=sample_rate,
        rate_hz=rate_hz,
        seed=seed,
        component_id=30,
        event_duration_s=(1.4, 3.6),
    )
    event_rng = component_rng(seed, _AMBIENCE_NAMESPACE, 4, sample_rate)
    for event in events:
        mix_at(
            result, _city_passby(event, sample_rate=sample_rate, rng=event_rng), event.start_sample
        )
    return result

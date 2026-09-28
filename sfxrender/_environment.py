"""Semantic wind, weather, fire, and wildlife effect compositions."""

from __future__ import annotations

import math

import numpy as np

from ._dsp import mix_at
from ._physics.rng import component_rng
from ._physics.stochastic import StochasticEvent, sample_stochastic_events
from ._physics.turbulence import render_turbulence_layer
from .types import FloatAudio

_ENVIRONMENT_NAMESPACE = 0x454E5649


def _raised_cosine(size: int) -> FloatAudio:
    if size <= 1:
        return np.ones(max(0, size), dtype=np.float32)
    position = np.linspace(0.0, 1.0, size, dtype=np.float32)
    return np.maximum(np.sin(np.float32(math.pi) * position), np.float32(0.0)) ** np.float32(1.3)


def render_wind(
    *, sample_rate: int, duration: float, intensity: str, texture: str, seed: int
) -> FloatAudio:
    size = max(1, round(duration * sample_rate))
    strong = intensity == "strong"
    leafy = texture == "leafy"
    result = render_turbulence_layer(
        size=size,
        sample_rate=sample_rate,
        seed=seed,
        component_id=10,
        low_hz=42.0,
        high_hz=520.0,
        amplitude=0.018 if strong else 0.012,
        variation_depth=0.36,
    )
    result += render_turbulence_layer(
        size=size,
        sample_rate=sample_rate,
        seed=seed,
        component_id=11,
        low_hz=300.0 if leafy else 500.0,
        high_hz=4_800.0 if leafy else 2_000.0,
        amplitude=0.012 if leafy else 0.004,
        variation_depth=0.48,
    )
    events = sample_stochastic_events(
        duration_s=duration,
        sample_rate=sample_rate,
        rate_hz=0.72 if strong else 0.28,
        seed=seed,
        component_id=12,
        event_duration_s=(0.8, 3.0),
    )
    for event in events:
        burst = render_turbulence_layer(
            size=event.duration_samples,
            sample_rate=sample_rate,
            seed=seed,
            component_id=13,
            event_index=event.event_index,
            low_hz=65.0,
            high_hz=4_600.0 if leafy else 1_800.0,
            amplitude=(0.030 if strong else 0.019) * event.level,
            variation_depth=0.32,
            envelope=_raised_cosine(event.duration_samples),
        )
        mix_at(result, burst, event.start_sample)
    return result


def render_transition_whoosh(
    *, sample_rate: int, duration: float, style: str, direction: str, seed: int
) -> FloatAudio:
    size = max(1, round(duration * sample_rate))
    sweep = np.linspace(0.0, 1.0, size, dtype=np.float32)
    envelope = _raised_cosine(size)
    brightness = sweep if direction == "rise" else np.float32(1.0) - sweep
    low_envelope = envelope * (np.float32(0.30) + np.float32(0.70) * (1.0 - brightness))
    high_envelope = envelope * (np.float32(0.25) + np.float32(0.75) * brightness)
    strength = 0.075 if style == "forceful" else 0.042
    result = render_turbulence_layer(
        size=size,
        sample_rate=sample_rate,
        seed=seed,
        component_id=20,
        low_hz=60.0,
        high_hz=680.0,
        amplitude=strength * 0.72,
        variation_depth=0.16,
        envelope=low_envelope,
    )
    result += render_turbulence_layer(
        size=size,
        sample_rate=sample_rate,
        seed=seed,
        component_id=21,
        low_hz=500.0,
        high_hz=5_500.0,
        amplitude=strength * 0.68,
        variation_depth=0.22,
        envelope=high_envelope,
    )
    return result


def _rain_drop(
    event: StochasticEvent,
    *,
    sample_rate: int,
    surface: str,
    seed: int,
) -> FloatAudio:
    size = event.duration_samples
    if surface == "window":
        low_hz, high_hz, gain = 900.0, 6_000.0, 0.015
    elif surface == "roof":
        low_hz, high_hz, gain = 250.0, 3_200.0, 0.019
    else:
        low_hz, high_hz, gain = 150.0, 1_800.0, 0.011
    time = np.arange(size, dtype=np.float32) / np.float32(sample_rate)
    decay_s = float(size) / sample_rate * 0.24
    envelope = np.exp(-time / np.float32(max(decay_s, 1.0 / sample_rate)))
    envelope *= np.minimum(time * np.float32(sample_rate * 140.0), np.float32(1.0))
    return render_turbulence_layer(
        size=size,
        sample_rate=sample_rate,
        seed=seed,
        component_id=30,
        event_index=event.event_index,
        low_hz=low_hz,
        high_hz=high_hz,
        amplitude=gain * event.level,
        variation_depth=0.12,
        envelope=envelope,
    )


def render_rain(
    *, sample_rate: int, duration: float, intensity: str, surface: str, seed: int
) -> FloatAudio:
    size = max(1, round(duration * sample_rate))
    rate_hz, level = {"light": (18.0, 0.65), "steady": (42.0, 1.0), "heavy": (82.0, 1.32)}[
        intensity
    ]
    result = render_turbulence_layer(
        size=size,
        sample_rate=sample_rate,
        seed=seed,
        component_id=31,
        low_hz=120.0,
        high_hz=6_200.0,
        amplitude=0.008 * level,
        variation_depth=0.22,
    )
    if intensity == "heavy":
        result += render_turbulence_layer(
            size=size,
            sample_rate=sample_rate,
            seed=seed,
            component_id=32,
            low_hz=45.0,
            high_hz=900.0,
            amplitude=0.010,
            variation_depth=0.28,
        )
    events = sample_stochastic_events(
        duration_s=duration,
        sample_rate=sample_rate,
        rate_hz=rate_hz,
        seed=seed,
        component_id=33,
        event_duration_s=(0.006, 0.045),
    )
    for event in events:
        mix_at(
            result,
            _rain_drop(event, sample_rate=sample_rate, surface=surface, seed=seed),
            event.start_sample,
        )
    return result


def _fire_pop(event: StochasticEvent, *, sample_rate: int, seed: int) -> FloatAudio:
    rng = component_rng(seed, _ENVIRONMENT_NAMESPACE, 41, event.event_index)
    bright = bool(rng.integers(0, 2))
    size = event.duration_samples
    time = np.arange(size, dtype=np.float32) / np.float32(sample_rate)
    decay_s = float(rng.uniform(0.006, 0.024))
    envelope = np.exp(-time / np.float32(decay_s))
    low_hz, high_hz = (800.0, 6_200.0) if bright else (120.0, 1_800.0)
    return render_turbulence_layer(
        size=size,
        sample_rate=sample_rate,
        seed=seed,
        component_id=42,
        event_index=event.event_index,
        low_hz=low_hz,
        high_hz=high_hz,
        amplitude=(0.026 if bright else 0.019) * event.level,
        variation_depth=0.08,
        envelope=envelope,
    )


def render_fire_crackle(
    *, sample_rate: int, duration: float, activity: str, seed: int
) -> FloatAudio:
    size = max(1, round(duration * sample_rate))
    result = render_turbulence_layer(
        size=size,
        sample_rate=sample_rate,
        seed=seed,
        component_id=40,
        low_hz=45.0,
        high_hz=620.0,
        amplitude=0.012,
        variation_depth=0.3,
    )
    result += render_turbulence_layer(
        size=size,
        sample_rate=sample_rate,
        seed=seed,
        component_id=41,
        low_hz=550.0,
        high_hz=5_200.0,
        amplitude=0.006,
        variation_depth=0.3,
    )
    events = sample_stochastic_events(
        duration_s=duration,
        sample_rate=sample_rate,
        rate_hz=2.4 if activity == "active" else 0.85,
        seed=seed,
        component_id=42,
        event_duration_s=(0.008, 0.075),
    )
    for event in events:
        mix_at(result, _fire_pop(event, sample_rate=sample_rate, seed=seed), event.start_sample)
    return result


def _bird_chirp(event: StochasticEvent, *, sample_rate: int, seed: int) -> FloatAudio:
    rng = component_rng(seed, _ENVIRONMENT_NAMESPACE, 51, event.event_index)
    output = np.zeros(event.duration_samples, dtype=np.float32)
    syllables = int(rng.integers(2, 5))
    syllable_samples = max(1, round(float(rng.uniform(0.045, 0.095)) * sample_rate))
    low_frequency = float(rng.uniform(1_650.0, 2_450.0))
    high_frequency = float(rng.uniform(2_400.0, 3_500.0))
    for syllable in range(syllables):
        start = round(syllable * output.size / syllables)
        end = min(output.size, start + syllable_samples)
        count = end - start
        if count < 4:
            continue
        position = np.linspace(0.0, 1.0, count, dtype=np.float32)
        time = np.arange(count, dtype=np.float32) / np.float32(sample_rate)
        phase = np.float32(2.0 * math.pi) * (
            low_frequency * time
            + (high_frequency - low_frequency) * np.square(time) / (2.0 * (count / sample_rate))
        )
        envelope = np.sin(np.float32(math.pi) * position) ** np.float32(2.0)
        output[start:end] += np.sin(phase) * envelope * np.float32(0.025 * event.level)
    return output


def render_birds_ambience(
    *, sample_rate: int, duration: float, activity: str, seed: int
) -> FloatAudio:
    size = max(1, round(duration * sample_rate))
    result = render_turbulence_layer(
        size=size,
        sample_rate=sample_rate,
        seed=seed,
        component_id=50,
        low_hz=700.0,
        high_hz=4_800.0,
        amplitude=0.0018,
        variation_depth=0.28,
    )
    events = sample_stochastic_events(
        duration_s=duration,
        sample_rate=sample_rate,
        rate_hz=0.72 if activity == "busy" else 0.32,
        seed=seed,
        component_id=51,
        event_duration_s=(0.25, 0.8),
    )
    for event in events:
        mix_at(result, _bird_chirp(event, sample_rate=sample_rate, seed=seed), event.start_sample)
    return result


def _cricket_chirp(event: StochasticEvent, *, sample_rate: int, seed: int) -> FloatAudio:
    rng = component_rng(seed, _ENVIRONMENT_NAMESPACE, 61, event.event_index)
    output = np.zeros(event.duration_samples, dtype=np.float32)
    high_frequency = min(5_200.0, sample_rate * 0.43)
    low_frequency = max(800.0, high_frequency * 0.72)
    frequency = float(rng.uniform(low_frequency, high_frequency))
    pulse_count = int(rng.integers(5, 10))
    pulse_size = max(4, round(0.012 * sample_rate))
    for pulse in range(pulse_count):
        start = round(pulse * output.size / pulse_count)
        end = min(output.size, start + pulse_size)
        count = end - start
        if count < 4:
            continue
        position = np.linspace(0.0, 1.0, count, dtype=np.float32)
        envelope = np.sin(np.float32(math.pi) * position) ** np.float32(2.0)
        phase = np.arange(count, dtype=np.float32) * np.float32(
            2.0 * math.pi * frequency / sample_rate
        )
        output[start:end] += np.sin(phase) * envelope * np.float32(0.022 * event.level)
    return output


def render_crickets_ambience(
    *, sample_rate: int, duration: float, activity: str, seed: int
) -> FloatAudio:
    size = max(1, round(duration * sample_rate))
    result = render_turbulence_layer(
        size=size,
        sample_rate=sample_rate,
        seed=seed,
        component_id=60,
        low_hz=1_800.0,
        high_hz=5_800.0,
        amplitude=0.0015,
        variation_depth=0.2,
    )
    events = sample_stochastic_events(
        duration_s=duration,
        sample_rate=sample_rate,
        rate_hz=1.35 if activity == "busy" else 0.52,
        seed=seed,
        component_id=61,
        event_duration_s=(0.18, 0.55),
    )
    for event in events:
        mix_at(
            result, _cricket_chirp(event, sample_rate=sample_rate, seed=seed), event.start_sample
        )
    return result

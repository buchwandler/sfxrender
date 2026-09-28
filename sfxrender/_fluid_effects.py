"""Turbulent water, thunder, and non-lexical crowd textures."""

from __future__ import annotations

import math

import numpy as np

from ._dsp import mix_at
from ._physics.modes import Mode, ModeSet
from ._physics.resonator import modal_response
from ._physics.rng import component_rng
from ._physics.stochastic import StochasticEvent, sample_stochastic_events
from ._physics.turbulence import render_turbulence_layer
from .types import FloatAudio

_FLUID_NAMESPACE = 0x464C5549


def _flow_envelope(size: int, sample_rate: int, rise_s: float, fall_s: float) -> FloatAudio:
    time = np.arange(size, dtype=np.float32) / np.float32(sample_rate)
    total_s = size / sample_rate
    attack = np.clip(time / np.float32(max(rise_s, 1.0 / sample_rate)), 0.0, 1.0)
    release = np.clip((total_s - time) / np.float32(max(fall_s, 1.0 / sample_rate)), 0.0, 1.0)
    return np.asarray(np.minimum(attack, release), dtype=np.float32)


def _vessel_modes(vessel: str, sample_rate: int, seed: int) -> ModeSet:
    specifications = {
        "glass": ((520.0, 0.36, 0.48), (1_180.0, 0.25, 0.30), (2_380.0, 0.16, 0.18)),
        "ceramic": ((340.0, 0.20, 0.52), (790.0, 0.15, 0.34), (1_620.0, 0.10, 0.20)),
        "metal": ((260.0, 0.45, 0.56), (630.0, 0.32, 0.37), (1_440.0, 0.22, 0.25)),
    }[vessel]
    rng = component_rng(seed, _FLUID_NAMESPACE, 1, sample_rate)
    return ModeSet(
        tuple(
            Mode(
                frequency * float(rng.uniform(0.985, 1.015)),
                decay,
                input_gain=gain,
                radiation_gain=gain,
                modal_mass_kg=0.06 + index * 0.035,
            )
            for index, (frequency, decay, gain) in enumerate(specifications)
            if frequency < sample_rate * 0.43
        )
    )


def _water_splash(
    event: StochasticEvent, *, sample_rate: int, strength: float, seed: int, component_id: int
) -> FloatAudio:
    size = event.duration_samples
    time = np.arange(size, dtype=np.float32) / np.float32(sample_rate)
    decay = float(size) / sample_rate * 0.32
    envelope = np.exp(-time / np.float32(max(decay, 1.0 / sample_rate)))
    envelope *= np.minimum(time * np.float32(sample_rate * 100.0), np.float32(1.0))
    return render_turbulence_layer(
        size=size,
        sample_rate=sample_rate,
        seed=seed,
        component_id=component_id,
        event_index=event.event_index,
        low_hz=350.0,
        high_hz=6_800.0,
        amplitude=strength * event.level,
        variation_depth=0.08,
        envelope=envelope,
    )


def render_water_pour(
    *, sample_rate: int, duration: float, flow: str, vessel: str, seed: int
) -> FloatAudio:
    size = max(1, round(duration * sample_rate))
    rate_hz, gain = {"trickle": (4.0, 0.55), "steady": (16.0, 0.9), "strong": (32.0, 1.25)}[flow]
    envelope = _flow_envelope(size, sample_rate, 0.12, 0.18)
    broad = render_turbulence_layer(
        size=size,
        sample_rate=sample_rate,
        seed=seed,
        component_id=10,
        low_hz=140.0,
        high_hz=5_800.0,
        amplitude=0.025 * gain,
        variation_depth=0.24,
        envelope=envelope,
    )
    low_flow = render_turbulence_layer(
        size=size,
        sample_rate=sample_rate,
        seed=seed,
        component_id=11,
        low_hz=45.0,
        high_hz=650.0,
        amplitude=0.010 * gain,
        variation_depth=0.3,
        envelope=envelope,
    )
    result = np.asarray(broad + low_flow, dtype=np.float32)
    events = sample_stochastic_events(
        duration_s=duration,
        sample_rate=sample_rate,
        rate_hz=rate_hz,
        seed=seed,
        component_id=12,
        event_duration_s=(0.008, 0.06),
    )
    for event in events:
        mix_at(
            result,
            _water_splash(
                event, sample_rate=sample_rate, strength=0.018 * gain, seed=seed, component_id=13
            ),
            event.start_sample,
        )
    resonant = modal_response(
        result,
        _vessel_modes(vessel, sample_rate, seed),
        sample_rate,
        tail_s=0.20,
    )
    output = np.zeros(max(result.size, resonant.size), dtype=np.float32)
    mix_at(output, result, 0)
    mix_at(output, resonant * np.float32(0.12), 0)
    return output


def render_water_running(*, sample_rate: int, duration: float, flow: str, seed: int) -> FloatAudio:
    size = max(1, round(duration * sample_rate))
    time = np.arange(size, dtype=np.float32) / np.float32(sample_rate)
    intensity = {"gentle": 0.52, "steady": 0.82, "strong": 1.2}[flow]
    control = np.float32(0.86) + np.float32(0.12) * np.sin(
        2.0 * math.pi * 0.23 * time
        + float(component_rng(seed, _FLUID_NAMESPACE, 20).uniform(-math.pi, math.pi))
    )
    envelope = np.asarray(control * _flow_envelope(size, sample_rate, 0.08, 0.10), dtype=np.float32)
    result = render_turbulence_layer(
        size=size,
        sample_rate=sample_rate,
        seed=seed,
        component_id=21,
        low_hz=90.0,
        high_hz=6_800.0,
        amplitude=0.032 * intensity,
        variation_depth=0.3,
        envelope=envelope,
    )
    events = sample_stochastic_events(
        duration_s=duration,
        sample_rate=sample_rate,
        rate_hz=4.0 if flow == "gentle" else 12.0 if flow == "steady" else 25.0,
        seed=seed,
        component_id=22,
        event_duration_s=(0.006, 0.03),
    )
    for event in events:
        mix_at(
            result,
            _water_splash(
                event,
                sample_rate=sample_rate,
                strength=0.008 * intensity,
                seed=seed,
                component_id=23,
            ),
            event.start_sample,
        )
    return result


def _thunder_crack(
    event: StochasticEvent,
    *,
    sample_rate: int,
    distance: str,
    strength: float,
    seed: int,
) -> FloatAudio:
    size = event.duration_samples
    time = np.arange(size, dtype=np.float32) / np.float32(sample_rate)
    rng = component_rng(seed, _FLUID_NAMESPACE, 34, event.event_index)
    decay = float(rng.uniform(0.018, 0.052))
    envelope = np.exp(-time / np.float32(decay))
    return render_turbulence_layer(
        size=size,
        sample_rate=sample_rate,
        seed=seed,
        component_id=35,
        event_index=event.event_index,
        low_hz=35.0,
        high_hz=900.0 if distance == "near" else 320.0,
        amplitude=strength * event.level,
        variation_depth=0.08,
        envelope=envelope,
    )


def render_thunder(
    *, sample_rate: int, duration: float, intensity: str, distance: str, seed: int
) -> FloatAudio:
    size = max(1, round(duration * sample_rate))
    time = np.arange(size, dtype=np.float32) / np.float32(sample_rate)
    tail_s = (2.7 if distance == "near" else 4.0) * (duration / 6.0)
    attack = np.clip(time / np.float32(0.035 if distance == "near" else 0.08), 0.0, 1.0)
    decay = np.exp(-time / np.float32(max(0.35, tail_s)))
    envelope = np.asarray(attack * decay, dtype=np.float32)
    strength = 1.0 if intensity == "strong" else 0.64
    low = render_turbulence_layer(
        size=size,
        sample_rate=sample_rate,
        seed=seed,
        component_id=30,
        low_hz=24.0,
        high_hz=180.0 if distance == "distant" else 360.0,
        amplitude=0.095 * strength,
        variation_depth=0.42,
        envelope=envelope,
    )
    crack = render_turbulence_layer(
        size=size,
        sample_rate=sample_rate,
        seed=seed,
        component_id=31,
        low_hz=100.0,
        high_hz=1_200.0 if distance == "near" else 520.0,
        amplitude=0.035 * strength,
        variation_depth=0.3,
        envelope=np.asarray(envelope * np.exp(-time / np.float32(0.45)), dtype=np.float32),
    )
    result = np.asarray(low + crack, dtype=np.float32)
    events = sample_stochastic_events(
        duration_s=min(duration, 1.4),
        sample_rate=sample_rate,
        rate_hz=2.1,
        seed=seed,
        component_id=32,
        event_duration_s=(0.035, 0.20),
    )
    for event in events:
        mix_at(
            result,
            _thunder_crack(
                event,
                sample_rate=sample_rate,
                distance=distance,
                strength=0.05 * strength,
                seed=seed,
            ),
            event.start_sample,
        )
    return result


def _crowd_grain(event: StochasticEvent, *, sample_rate: int, seed: int) -> FloatAudio:
    size = event.duration_samples
    position = np.linspace(0.0, 1.0, size, dtype=np.float32)
    envelope = np.maximum(np.sin(np.float32(math.pi) * position), np.float32(0.0)) ** np.float32(
        1.1
    )
    result = np.zeros(size, dtype=np.float32)
    bands = ((110.0, 430.0, 0.010), (300.0, 1_100.0, 0.008), (750.0, 2_300.0, 0.004))
    for index, (low, high, gain) in enumerate(bands):
        grain = render_turbulence_layer(
            size=size,
            sample_rate=sample_rate,
            seed=seed,
            component_id=40 + index,
            event_index=event.event_index,
            low_hz=low,
            high_hz=high,
            amplitude=gain * event.level,
            variation_depth=0.48,
            envelope=envelope,
        )
        result += grain
    return result


def render_crowd_murmur(
    *, sample_rate: int, duration: float, density: str, seed: int
) -> FloatAudio:
    size = max(1, round(duration * sample_rate))
    base_level = {"sparse": 0.004, "moderate": 0.007, "busy": 0.010}[density]
    bed = render_turbulence_layer(
        size=size,
        sample_rate=sample_rate,
        seed=seed,
        component_id=50,
        low_hz=85.0,
        high_hz=680.0,
        amplitude=base_level,
        variation_depth=0.24,
    )
    rate_hz = {"sparse": 7.0, "moderate": 14.0, "busy": 25.0}[density]
    events = sample_stochastic_events(
        duration_s=duration,
        sample_rate=sample_rate,
        rate_hz=rate_hz,
        seed=seed,
        component_id=51,
        event_duration_s=(0.18, 0.62),
    )
    for event in events:
        mix_at(bed, _crowd_grain(event, sample_rate=sample_rate, seed=seed), event.start_sample)
    return bed

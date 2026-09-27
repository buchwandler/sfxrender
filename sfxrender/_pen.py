"""Deterministic close-mic pen-on-paper synthesis."""

from __future__ import annotations

import math

import numpy as np

from ._dsp import asymmetric_pulse, bandpass_noise
from ._physics.friction import (
    LuGrePreset,
    lugre_friction,
    make_roughness_profile,
    roughness_velocity,
)
from .types import FloatAudio

_PEN_SALT = 0x50454E57
_STROKE_TIMING = 1
_PAPER_ROUGHNESS = 2
_FRICTION_TEXTURE = 3
_LIFT_CONTACTS = 4
_BODY_RESONANCE = 5
_PEN_SPEED_M_S = {"slow": 0.035, "normal": 0.060, "fast": 0.090}
_PEN_FRICTION = LuGrePreset(
    static_coefficient=0.34,
    dynamic_coefficient=0.26,
    stribeck_velocity_m_s=0.025,
    bristle_stiffness_n_m=45_000.0,
    bristle_damping_n_s_m=0.10,
    viscous_coefficient_n_s_m=0.03,
)


def _component_rng(seed: int, event_index: int, component_id: int) -> np.random.Generator:
    """Return a stable stream independent of other pen-synthesis layers."""
    sequence = np.random.SeedSequence([int(seed), _PEN_SALT, event_index, component_id])
    return np.random.default_rng(sequence)


def _stroke_controls(
    *, sample_rate: int, duration: float, speed: str, pressure: float, seed: int
) -> tuple[FloatAudio, FloatAudio, FloatAudio, FloatAudio, np.ndarray]:
    size = max(1, round(duration * sample_rate))
    timing_rng = _component_rng(seed, 0, _STROKE_TIMING)
    velocity = np.zeros(size, dtype=np.float32)
    load = np.zeros(size, dtype=np.float32)
    activity = np.zeros(size, dtype=np.float32)
    position = np.zeros(size, dtype=np.float32)
    tap_samples: list[int] = []
    base_speed = _PEN_SPEED_M_S[speed]
    normal_load = 0.42 + 2.30 * pressure
    cursor = 0
    distance = 0.0
    stroke_index = 0
    while cursor < size:
        requested_length = float(timing_rng.uniform(0.008, 0.035))
        local_speed = base_speed * float(timing_rng.uniform(0.82, 1.18))
        stroke_frames = max(3, round(requested_length / local_speed * sample_rate))
        end = min(size, cursor + stroke_frames)
        count = end - cursor
        phase = np.linspace(0.0, 1.0, count, dtype=np.float32)
        edge = np.maximum(0.0, np.sin(np.float32(math.pi) * phase)) ** np.float32(0.42)
        rate = float(timing_rng.uniform(2.0, 5.5))
        wobble_phase = float(timing_rng.uniform(-math.pi, math.pi))
        local_time = np.arange(count, dtype=np.float32) / np.float32(sample_rate)
        phrasing = 0.78 + 0.22 * np.sin(2.0 * math.pi * rate * local_time + wobble_phase)
        local_activity = np.asarray(edge * phrasing, dtype=np.float32)
        direction = -1.0 if stroke_index % 3 == 2 else 1.0
        local_velocity = direction * local_speed * (0.78 + 0.22 * phrasing)
        velocity[cursor:end] = local_velocity
        load[cursor:end] = np.float32(normal_load) * local_activity
        activity[cursor:end] = local_activity
        travelled = np.linspace(0.0, requested_length, count, dtype=np.float32)
        position[cursor:end] = np.float32(distance) + travelled
        distance += requested_length
        tap_samples.append(cursor)
        stroke_index += 1
        if end >= size:
            break
        gap = float(timing_rng.uniform(0.015, 0.045))
        cursor = min(size, end + max(1, round(gap * sample_rate)))
    return velocity, load, activity, position, np.asarray(tap_samples, dtype=np.int64)


def render_pen_write(
    *, sample_rate: int, duration: float, speed: str, pressure: float, seed: int
) -> FloatAudio:
    """Render dry handwriting with seeded strokes, friction, lifts, and tiny taps."""
    relative_velocity, normal_load, activity, position, tap_samples = _stroke_controls(
        sample_rate=sample_rate,
        duration=duration,
        speed=speed,
        pressure=pressure,
        seed=seed,
    )
    size = relative_velocity.size
    rough_rng = _component_rng(seed, 0, _PAPER_ROUGHNESS)
    friction_rng = _component_rng(seed, 0, _FRICTION_TEXTURE)
    contact_rng = _component_rng(seed, 0, _LIFT_CONTACTS)
    body_rng = _component_rng(seed, 0, _BODY_RESONANCE)
    path_length = max(0.04, float(np.max(position)) + 0.025)
    roughness = make_roughness_profile(
        roughness_rms_m=4.8e-6,
        correlation_length_m=0.00042,
        seed=int(rough_rng.integers(0, 2**32, dtype=np.uint32)),
        length_m=path_length,
    )
    rough_velocity = roughness_velocity(roughness, position, sample_rate)
    trace = lugre_friction(
        relative_velocity,
        normal_load,
        _PEN_FRICTION,
        sample_rate,
    )
    power_texture = np.sqrt(np.clip(trace.power_w / np.float32(0.07), 0.0, 1.6))
    release_texture = np.sqrt(
        np.clip(
            trace.release_energy_j * sample_rate / np.float32(0.09),
            0.0,
            1.0,
        )
    )
    rough_texture = 1.0 + 0.25 * np.tanh(np.abs(rough_velocity) / np.float32(0.025))
    pressure_gain = np.sqrt(np.clip(normal_load / np.float32(1.6), 0.0, 2.0))
    friction_envelope = activity * (0.55 + 0.45 * power_texture + 0.18 * release_texture)
    friction_envelope *= pressure_gain * rough_texture

    speed_ratio = _PEN_SPEED_M_S[speed] / _PEN_SPEED_M_S["normal"]
    high = bandpass_noise(
        friction_rng,
        size,
        sample_rate,
        min(700.0, sample_rate * 0.18),
        min(6_500.0, sample_rate * 0.44),
    )
    mid = bandpass_noise(
        friction_rng,
        size,
        sample_rate,
        320.0,
        min(2_600.0, sample_rate * 0.44),
    )
    low = bandpass_noise(
        body_rng,
        size,
        sample_rate,
        120.0,
        min(1_200.0, sample_rate * 0.40),
    )
    scratch = high * np.float32(0.78 + 0.10 * speed_ratio) + mid * np.float32(0.34)
    scratch *= friction_envelope * np.float32(0.105)
    body = low * activity * pressure_gain * np.float32(0.012)

    contacts = np.zeros(size, dtype=np.float32)
    for index, start in enumerate(tap_samples):
        if index == 0 or start >= size:
            continue
        tap_size = min(size - int(start), max(1, round(0.018 * sample_rate)))
        pulse = asymmetric_pulse(
            tap_size,
            sample_rate,
            attack_s=0.001,
            decay_s=0.006 + float(contact_rng.uniform(0.0, 0.004)),
            amplitude=float(contact_rng.uniform(0.10, 0.18)),
        )
        contacts[int(start) : int(start) + tap_size] += pulse
    contact_noise = bandpass_noise(
        contact_rng, size, sample_rate, 400.0, min(4_800.0, sample_rate * 0.44)
    )
    body += contact_noise * contacts * np.float32(0.06)
    return np.asarray(scratch + body, dtype=np.float32)

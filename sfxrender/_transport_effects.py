"""Elevator and vehicle action compositions."""

from __future__ import annotations

import math

import numpy as np

from ._device_effects import render_device_beep
from ._dsp import mix_at
from ._object_effects import render_switch_toggle
from ._physics.closure import render_terminal_closure
from ._physics.contact import ImpactContact
from ._physics.electromechanical import render_electromechanical_hum
from ._physics.models import FrictionProfile, MotionCurve
from ._physics.modes import Mode, ModeSet
from ._physics.rotating import render_rotating_machine
from ._physics.sliding import render_sliding_source
from ._physics.turbulence import render_turbulence_layer
from .types import FloatAudio


def _motion(size: int, sample_rate: int) -> MotionCurve:
    position = np.linspace(0.0, 1.0, size, dtype=np.float32)
    position = position * position * (np.float32(3.0) - np.float32(2.0) * position)
    velocity = np.gradient(position).astype(np.float32) * np.float32(sample_rate)
    acceleration = np.gradient(velocity).astype(np.float32) * np.float32(sample_rate)
    return MotionCurve(position, velocity, acceleration)


def render_elevator_arrive(
    *, sample_rate: int, duration: float, size: str, chime: str, seed: int
) -> FloatAudio:
    size_samples = max(1, round(duration * sample_rate))
    time = np.arange(size_samples, dtype=np.float32) / np.float32(sample_rate)
    progress = np.linspace(0.0, 1.0, size_samples, dtype=np.float32)
    initial_speed = 8.5 if size == "small" else 6.2
    speed_curve = np.asarray(
        initial_speed * (1.0 - progress) ** np.float32(1.5) + 0.3,
        dtype=np.float32,
    )
    decay = np.exp(-time / np.float32(max(0.25, duration * 0.52))).astype(np.float32)
    motor = (
        render_rotating_machine(
            speed_curve_hz=speed_curve,
            sample_rate=sample_rate,
            seed=seed,
            event_index=1,
            ripple_rate_hz=9.0,
            ripple_depth=0.22,
            amplitude=0.045 if size == "large" else 0.032,
            harmonic_gains=(1.0, 0.48, 0.25, 0.14, 0.08),
        )
        * decay
    )
    hum = render_electromechanical_hum(
        sample_rate=sample_rate,
        size=size_samples,
        seed=seed,
        event_index=2,
        fundamental_hz=60.0,
        amplitude=0.0035 if size == "large" else 0.0025,
        envelope=decay,
        harmonic_gains=(1.0, 0.26, 0.12, 0.05),
    )
    result = np.zeros(size_samples + round(0.22 * sample_rate), dtype=np.float32)
    mix_at(result, motor, 0)
    mix_at(result, hum, 0)
    ding = render_device_beep(
        sample_rate=sample_rate,
        style="soft",
        pattern="double" if chime == "double" else "single",
        seed=seed,
    )
    mix_at(result, ding, round(0.12 * sample_rate))

    slide_size = max(1, round(0.56 * sample_rate))
    slider = render_sliding_source(
        motion=_motion(slide_size, sample_rate),
        profile=FrictionProfile(
            base_gain=0.11 if size == "large" else 0.08,
            roughness=0.36,
            noise_band_hz=(180.0, 3_000.0),
            stick_strength=0.12,
            slip_strength=0.20,
            f0_hz=(180.0, 540.0),
            harmonic_rolloff=(1.4, 2.1),
            chaos_amount=0.14,
        ),
        sample_rate=sample_rate,
        seed=seed,
        event_index=3,
    )
    mix_at(result, slider, round(min(duration * 0.55, 0.8) * sample_rate))
    terminal = render_terminal_closure(
        contact=ImpactContact(0.018, 2.5e7, exponent=1.5, restitution=0.20),
        velocity_m_s=0.22 if size == "small" else 0.30,
        modes=ModeSet(
            (
                Mode(240.0, 0.10, input_gain=0.52, radiation_gain=0.24, modal_mass_kg=0.16),
                Mode(510.0, 0.075, input_gain=0.36, radiation_gain=0.18, modal_mass_kg=0.09),
                Mode(1_120.0, 0.045, input_gain=0.22, radiation_gain=0.12, modal_mass_kg=0.05),
            )
        ),
        sample_rate=sample_rate,
        seed=seed,
        event_index=4,
        output_gain=12.0,
        microscopic_gain=0.1,
        tail_s=0.18,
    )
    mix_at(result, terminal, size_samples - round(0.02 * sample_rate))
    return result


def _engine_speed_curve(*, size: int, sample_rate: int, action: str, vehicle: str) -> FloatAudio:
    progress = np.linspace(0.0, 1.0, size, dtype=np.float32)
    base = 22.0 if vehicle == "compact" else 16.0
    if action == "start":
        speed = 5.0 + (base - 5.0) * np.clip(progress / np.float32(0.42), 0.0, 1.0)
    elif action == "rev":
        speed = base + (33.0 if vehicle == "compact" else 25.0) * np.sin(
            np.float32(math.pi) * progress
        )
    else:
        speed = base + 0.8 * np.sin(2.0 * math.pi * 0.72 * progress)
    return np.asarray(np.clip(speed, 0.0, sample_rate * 0.20), dtype=np.float32)


def _engine_bed(
    *, sample_rate: int, duration: float, action: str, vehicle: str, seed: int
) -> FloatAudio:
    size = max(1, round(duration * sample_rate))
    speed = _engine_speed_curve(size=size, sample_rate=sample_rate, action=action, vehicle=vehicle)
    amplitude = 0.12 if vehicle == "compact" else 0.16
    engine = render_rotating_machine(
        speed_curve_hz=speed,
        sample_rate=sample_rate,
        seed=seed,
        event_index=10,
        ripple_rate_hz=18.0 if vehicle == "compact" else 12.0,
        ripple_depth=0.24,
        amplitude=amplitude,
        harmonic_gains=(1.0, 0.62, 0.40, 0.27, 0.18, 0.11),
    )
    time = np.arange(size, dtype=np.float32) / np.float32(sample_rate)
    envelope = np.full(size, 1.0, dtype=np.float32)
    if action == "start":
        envelope *= np.clip(time / np.float32(0.18), 0.0, 1.0)
    exhaust = render_turbulence_layer(
        size=size,
        sample_rate=sample_rate,
        seed=seed,
        component_id=11,
        low_hz=45.0,
        high_hz=720.0 if vehicle == "compact" else 480.0,
        amplitude=0.014 if vehicle == "compact" else 0.021,
        variation_depth=0.24,
        envelope=envelope,
    )
    output = np.asarray(engine * envelope + exhaust, dtype=np.float32)
    if action == "start":
        mix_at(
            output,
            render_switch_toggle(sample_rate=sample_rate, state="on", seed=seed) * np.float32(0.24),
            0,
        )
    return output


def render_car_engine(
    *, sample_rate: int, duration: float, action: str, vehicle: str, seed: int
) -> FloatAudio:
    return _engine_bed(
        sample_rate=sample_rate,
        duration=duration,
        action=action,
        vehicle=vehicle,
        seed=seed,
    )


def render_car_passby(
    *, sample_rate: int, duration: float, speed: str, vehicle: str, seed: int
) -> FloatAudio:
    size = max(1, round(duration * sample_rate))
    position = np.linspace(0.0, 1.0, size, dtype=np.float32)
    envelope = np.maximum(np.sin(np.float32(math.pi) * position), np.float32(0.0)) ** np.float32(
        0.72
    )
    peak_hz = (50.0 if speed == "fast" else 34.0) * (1.0 if vehicle == "compact" else 0.78)
    speed_curve = 8.0 + peak_hz * np.exp(-np.square((position - 0.50) / 0.25))
    engine = render_rotating_machine(
        speed_curve_hz=np.asarray(speed_curve, dtype=np.float32),
        sample_rate=sample_rate,
        seed=seed,
        event_index=20,
        ripple_rate_hz=16.0,
        ripple_depth=0.2,
        amplitude=0.13 if vehicle == "compact" else 0.17,
        harmonic_gains=(1.0, 0.60, 0.38, 0.24, 0.14, 0.08),
    )
    road = render_turbulence_layer(
        size=size,
        sample_rate=sample_rate,
        seed=seed,
        component_id=21,
        low_hz=35.0,
        high_hz=1_000.0,
        amplitude=0.022 if speed == "fast" else 0.015,
        variation_depth=0.18,
        envelope=envelope,
    )
    return np.asarray(engine * envelope + road, dtype=np.float32)

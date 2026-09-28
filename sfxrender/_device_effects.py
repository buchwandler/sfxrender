"""Semantic electronic device, notification, and vibration effects."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np

from ._dsp import mix_at
from ._electronic import TonePreset, TonePulse, render_tone_pattern
from ._object_effects import render_switch_toggle
from ._physics.electromechanical import render_electromechanical_hum
from ._physics.geometry import BoundaryCondition, RectangularPlate, rectangular_plate_modes
from ._physics.modes import Mode, ModeSet
from ._physics.presets import FLOOR_OBJECTS
from ._physics.resonator import modal_response
from ._physics.rng import component_rng
from ._physics.rotating import render_rotating_machine
from .types import FloatAudio

_DEVICE_NAMESPACE = 0x44455643
_HUM_PRESETS = {
    "transformer": (60.0, 0.021, (1.0, 0.32, 0.19, 0.11, 0.06)),
    "appliance": (60.0, 0.018, (0.82, 0.44, 0.24, 0.14, 0.08)),
}
_DEVICE_PRESETS = {
    "small": (52.0, 0.014, (1.0, 0.42, 0.20, 0.08), 1.12, (1.0, 0.42, 0.20, 0.08)),
    "appliance": (36.0, 0.010, (1.0, 0.55, 0.28, 0.14, 0.07), 1.72, (1.0, 0.55, 0.28, 0.14, 0.07)),
}


def _hum_parameters(device: str) -> tuple[float, float, tuple[float, ...]]:
    return _HUM_PRESETS[device]


def render_electronics_hum(
    *, sample_rate: int, duration: float, device: str, seed: int
) -> FloatAudio:
    fundamental, amplitude, harmonics = _hum_parameters(device)
    size = max(1, round(duration * sample_rate))
    rng = component_rng(seed, _DEVICE_NAMESPACE, 1, sample_rate)
    time = np.arange(size, dtype=np.float32) / np.float32(sample_rate)
    phase = float(rng.uniform(-math.pi, math.pi))
    envelope = 0.93 + 0.035 * np.sin(2.0 * math.pi * 0.7 * time + phase)
    return render_electromechanical_hum(
        sample_rate=sample_rate,
        size=size,
        seed=seed,
        event_index=2,
        fundamental_hz=fundamental,
        amplitude=amplitude,
        envelope=np.asarray(envelope, dtype=np.float32),
        harmonic_gains=harmonics,
    )


def _tone_preset(style: str) -> TonePreset:
    if style in {"soft", "gentle", "confirmation"}:
        return TonePreset(
            frequency_hz=880.0,
            harmonic_gains=(1.0, 0.22, 0.08),
            attack_s=0.004,
            release_s=0.035,
            bandwidth_hz=(420.0, 4_200.0),
            transient_click=0.012,
        )
    return TonePreset(
        frequency_hz=1_060.0,
        harmonic_gains=(0.94, 0.35, 0.15),
        attack_s=0.002,
        release_s=0.045,
        bandwidth_hz=(520.0, 4_800.0),
        distortion=0.025,
        transient_click=0.024,
    )


def render_device_beep(*, sample_rate: int, style: str, pattern: str, seed: int) -> FloatAudio:
    count = {"single": 1, "double": 2, "triple": 3}[pattern]
    interval = 0.19
    note_duration = 0.095
    total_duration = interval * (count - 1) + note_duration
    level = 0.62 if style == "soft" else 0.72
    pitch_step = 1.0 if style == "soft" else 0.93
    pulses = tuple(
        TonePulse(
            start_s=index * interval,
            duration_s=note_duration,
            level=level,
            pitch_scale=pitch_step**index,
        )
        for index in range(count)
    )
    return render_tone_pattern(
        preset=_tone_preset(style),
        pulses=pulses,
        duration_s=total_duration,
        sample_rate=sample_rate,
        rng=component_rng(seed, _DEVICE_NAMESPACE, 3, sample_rate),
    )


def render_phone_notification(*, sample_rate: int, style: str, count: int, seed: int) -> FloatAudio:
    preset = _tone_preset("gentle" if style == "gentle" else "urgent")
    note_starts, pitch_scales = (
        ((0.0, 0.13, 0.29), (1.0, 1.25, 1.5))
        if style == "gentle"
        else ((0.0, 0.105, 0.235), (1.45, 1.0, 1.65))
    )
    note_duration = 0.12 if style == "gentle" else 0.105
    cycle_duration = note_starts[-1] + note_duration + 0.18
    pulses = tuple(
        TonePulse(
            start_s=cycle * cycle_duration + start,
            duration_s=note_duration,
            level=(0.58 if style == "gentle" else 0.68) * (1.0 - 0.08 * note),
            pitch_scale=pitch_scales[note],
        )
        for cycle in range(count)
        for note, start in enumerate(note_starts)
    )
    duration = (count - 1) * cycle_duration + note_starts[-1] + note_duration
    return render_tone_pattern(
        preset=preset,
        pulses=pulses,
        duration_s=duration,
        sample_rate=sample_rate,
        rng=component_rng(seed, _DEVICE_NAMESPACE, 4, sample_rate),
    )


def _phone_modes(*, sample_rate: int, surface: str, seed: int) -> ModeSet:
    rng = component_rng(seed, _DEVICE_NAMESPACE, 5, sample_rate)
    body = tuple(
        Mode(
            frequency * float(rng.uniform(0.98, 1.02)),
            decay,
            input_gain=gain,
            radiation_gain=gain * 0.52,
            modal_mass_kg=0.012 + index * 0.006,
        )
        for index, (frequency, decay, gain) in enumerate(
            ((185.0, 0.055, 0.64), (420.0, 0.045, 0.48), (980.0, 0.032, 0.28))
        )
        if frequency < sample_rate * 0.43
    )
    material = FLOOR_OBJECTS[surface].material
    tread = RectangularPlate(
        0.88 if surface == "wood" else 0.62,
        0.52 if surface == "wood" else 0.42,
        0.026 if surface == "wood" else 0.032,
        BoundaryCondition.FLOOR_SUPPORTED,
    )
    plate = rectangular_plate_modes(
        tread,
        material,
        sample_rate=sample_rate,
        max_modes=24,
        contact_x=float(rng.uniform(0.28, 0.72)),
        contact_y=float(rng.uniform(0.28, 0.72)),
    )
    surface_modes = tuple(
        replace(
            mode,
            decay_s=mode.decay_s * 0.62,
            input_gain=mode.input_gain * 0.18,
            radiation_gain=mode.radiation_gain * 0.42,
            modal_mass_kg=mode.modal_mass_kg * 2.4,
        )
        for mode in plate.modes
    )
    return ModeSet(body + surface_modes)


def render_phone_vibrate(
    *,
    sample_rate: int,
    duration: float,
    intensity: str,
    pattern: str,
    surface: str,
    seed: int,
) -> FloatAudio:
    size = max(1, round(duration * sample_rate))
    time = np.arange(size, dtype=np.float32) / np.float32(sample_rate)
    if pattern == "steady":
        activity = np.ones(size, dtype=np.float32)
    else:
        period = round(0.42 * sample_rate)
        activity = ((np.arange(size) % period) < round(0.24 * sample_rate)).astype(np.float32)
    fade = np.clip(np.minimum(time / 0.018, (duration - time) / 0.028), 0.0, 1.0)
    activity *= fade.astype(np.float32)
    level = {"gentle": 0.55, "normal": 1.0, "strong": 1.5}[intensity]
    base_speed = 142.0 if intensity != "strong" else 158.0
    speed_curve = activity * np.float32(base_speed)
    motor = render_rotating_machine(
        speed_curve_hz=speed_curve,
        sample_rate=sample_rate,
        seed=seed,
        event_index=6,
        ripple_rate_hz=18.0,
        ripple_depth=0.24,
        amplitude=0.034 * level,
        harmonic_gains=(1.0, 0.48, 0.23, 0.12),
    )
    phase = np.zeros(size, dtype=np.float64)
    if size > 1:
        phase[1:] = np.cumsum(2.0 * math.pi * speed_curve[:-1] / sample_rate)
    force = np.asarray(np.sin(phase) * activity * (0.028 * level), dtype=np.float32)
    modes = _phone_modes(sample_rate=sample_rate, surface=surface, seed=seed)
    resonant = modal_response(force, modes, sample_rate, tail_s=0.10)
    output = np.zeros(max(motor.size, resonant.size), dtype=np.float32)
    mix_at(output, motor, 0)
    mix_at(output, resonant * np.float32(12.0), 0)
    return output


def _device_motor(*, sample_rate: int, device: str, seed: int, envelope: FloatAudio) -> FloatAudio:
    speed_hz, _hum_amplitude, _harmonics, _duration, motor_harmonics = _DEVICE_PRESETS[device]
    speed_curve = envelope * np.float32(speed_hz)
    return render_rotating_machine(
        speed_curve_hz=speed_curve,
        sample_rate=sample_rate,
        seed=seed,
        event_index=7,
        ripple_rate_hz=13.0 if device == "small" else 9.0,
        ripple_depth=0.18,
        amplitude=0.032 if device == "small" else 0.045,
        harmonic_gains=motor_harmonics,
    )


def _device_hum(
    *, sample_rate: int, size: int, device: str, seed: int, envelope: FloatAudio
) -> FloatAudio:
    _speed_hz, amplitude, harmonics, _duration, _motor_harmonics = _DEVICE_PRESETS[device]
    return render_electromechanical_hum(
        sample_rate=sample_rate,
        size=size,
        seed=seed,
        event_index=8,
        fundamental_hz=60.0,
        amplitude=amplitude,
        envelope=envelope,
        harmonic_gains=harmonics,
    )


def render_device_power_on(*, sample_rate: int, device: str, seed: int) -> FloatAudio:
    _speed_hz, _amplitude, _harmonics, duration, _motor_harmonics = _DEVICE_PRESETS[device]
    size = round(duration * sample_rate)
    time = np.arange(size, dtype=np.float32) / np.float32(sample_rate)
    ramp_start = 0.12 if device == "small" else 0.18
    ramp = np.clip((time - np.float32(ramp_start)) / np.float32(0.36), 0.0, 1.0)
    envelope = np.asarray(
        ramp * (0.92 + 0.04 * np.sin(2.0 * math.pi * 0.6 * time)), dtype=np.float32
    )
    motor = _device_motor(sample_rate=sample_rate, device=device, seed=seed, envelope=envelope)
    hum = _device_hum(
        sample_rate=sample_rate, size=size, device=device, seed=seed, envelope=envelope
    )
    confirmation = render_device_beep(
        sample_rate=sample_rate, style="soft", pattern="single", seed=seed
    )
    switch = render_switch_toggle(sample_rate=sample_rate, state="on", seed=seed)
    output = np.zeros(size, dtype=np.float32)
    mix_at(output, switch, 0)
    mix_at(output, motor, 0)
    mix_at(output, hum, 0)
    tone_start = round((duration - 0.25) * sample_rate)
    mix_at(output, confirmation * np.float32(0.72), tone_start)
    return output


def render_device_power_off(*, sample_rate: int, device: str, seed: int) -> FloatAudio:
    _speed_hz, _amplitude, _harmonics, duration, _motor_harmonics = _DEVICE_PRESETS[device]
    size = round(duration * sample_rate)
    time = np.arange(size, dtype=np.float32) / np.float32(sample_rate)
    envelope = np.exp(-time / np.float32(0.34 if device == "small" else 0.48)).astype(np.float32)
    motor = _device_motor(sample_rate=sample_rate, device=device, seed=seed, envelope=envelope)
    hum = _device_hum(
        sample_rate=sample_rate, size=size, device=device, seed=seed, envelope=envelope
    )
    switch = render_switch_toggle(sample_rate=sample_rate, state="off", seed=seed)
    output = np.zeros(size, dtype=np.float32)
    mix_at(output, motor, 0)
    mix_at(output, hum, 0)
    mix_at(output, switch, max(0, size - switch.size))
    return output

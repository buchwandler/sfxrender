"""Semantic friction, fabric, and structural movement effects."""

from __future__ import annotations

import math

import numpy as np

from ._dsp import mix_at
from ._physics.closure import render_terminal_closure
from ._physics.contact import ImpactContact
from ._physics.geometry import rectangular_plate_modes
from ._physics.models import FrictionProfile, ModalBody, MotionCurve
from ._physics.models import Mode as BodyMode
from ._physics.modes import Mode, ModeSet
from ._physics.presets import FLOOR_OBJECTS
from ._physics.rng import component_rng
from ._physics.sliding import render_sliding_source
from ._physics.stochastic import sample_stochastic_events
from ._physics.thin_material import ThinMaterialPreset, render_thin_material_source
from .types import FloatAudio

_MATERIAL_NAMESPACE = 0x4D415445


def _motion_curve(*, size: int, sample_rate: int, reverse: bool = False) -> MotionCurve:
    position = np.linspace(0.0, 1.0, size, dtype=np.float32)
    smooth = position * position * (np.float32(3.0) - np.float32(2.0) * position)
    if reverse:
        smooth = np.float32(1.0) - smooth
    velocity = np.gradient(smooth).astype(np.float32) * np.float32(sample_rate)
    acceleration = np.gradient(velocity).astype(np.float32) * np.float32(sample_rate)
    return MotionCurve(smooth, velocity, acceleration)


def _floor_body(
    *, surface: str, sample_rate: int, seed: int, component_id: int, max_modes: int = 18
) -> ModalBody:
    preset = FLOOR_OBJECTS[surface]
    rng = component_rng(seed, _MATERIAL_NAMESPACE, component_id, sample_rate)
    modes = rectangular_plate_modes(
        preset.geometry,
        preset.material,
        sample_rate=sample_rate,
        max_modes=max_modes,
        contact_x=float(rng.uniform(0.34, 0.66)),
        contact_y=float(rng.uniform(0.34, 0.66)),
    )
    body_modes = tuple(
        BodyMode(
            mode.frequency_hz,
            mode.decay_s,
            gain=abs(mode.input_gain * mode.radiation_gain),
        )
        for mode in modes.modes
    )
    return ModalBody(
        body_modes,
        radiation_gain=preset.modal_gain,
        damping_scale=preset.mode_decay_scale,
    )


def _modal_body(modes: ModeSet, *, radiation_gain: float, damping_scale: float = 1.0) -> ModalBody:
    body_modes = tuple(
        BodyMode(
            mode.frequency_hz,
            mode.decay_s,
            gain=abs(mode.input_gain * mode.radiation_gain),
        )
        for mode in modes.modes
    )
    return ModalBody(
        body_modes,
        radiation_gain=radiation_gain,
        damping_scale=damping_scale,
    )


def _resonator_modes(body: ModalBody) -> ModeSet:
    return ModeSet(
        tuple(
            Mode(
                mode.frequency_hz,
                mode.decay_s,
                input_gain=mode.gain,
                radiation_gain=body.radiation_gain,
                modal_mass_kg=1.0,
            )
            for mode in body.modes
        )
    )


def render_chair_move(
    *, sample_rate: int, duration: float, surface: str, effort: str, action: str, seed: int
) -> FloatAudio:
    size = max(1, round(duration * sample_rate))
    motion = _motion_curve(size=size, sample_rate=sample_rate)
    body = _floor_body(surface=surface, sample_rate=sample_rate, seed=seed, component_id=1)
    surface_gain = {"wood": 1.0, "stone": 1.25, "carpet": 0.38}[surface]
    effort_gain = {"light": 0.72, "firm": 1.0}[effort]
    profile = FrictionProfile(
        base_gain=0.18 * surface_gain * effort_gain,
        roughness=0.64,
        noise_band_hz=(55.0, 3_200.0),
        stick_strength=0.24 * effort_gain,
        slip_strength=0.30 * surface_gain,
        f0_hz=(45.0, 125.0),
        harmonic_rolloff=(1.3, 2.1),
        chaos_amount=0.34,
    )
    result = render_sliding_source(
        motion=motion,
        profile=profile,
        sample_rate=sample_rate,
        seed=seed,
        event_index=2,
        body=body,
    )
    if action == "set_down":
        modes = _resonator_modes(body)
        contact = ImpactContact(0.11, 1.1e6, exponent=1.5, restitution=0.20)
        closure = render_terminal_closure(
            contact=contact,
            velocity_m_s=0.20 if effort == "light" else 0.34,
            modes=modes,
            sample_rate=sample_rate,
            seed=seed,
            event_index=3,
            output_gain=10.0,
            microscopic_gain=0.08,
            tail_s=0.28,
        )
        closure_start = size - round(0.08 * sample_rate)
        output = np.zeros(max(result.size, closure_start + closure.size), dtype=np.float32)
        mix_at(output, result, 0)
        mix_at(output, closure, closure_start)
        return output
    return result


def render_lock_turn(
    *, sample_rate: int, duration: float, style: str, force: str, seed: int
) -> FloatAudio:
    size = max(1, round(duration * sample_rate))
    heavy = style == "deadbolt"
    force_scale = 1.0 if force == "firm" else 0.72
    modes = ModeSet(
        (
            Mode(
                280.0 if heavy else 430.0,
                0.10,
                input_gain=0.56,
                radiation_gain=0.22,
                modal_mass_kg=0.035,
            ),
            Mode(
                690.0 if heavy else 1_080.0,
                0.075,
                input_gain=0.40,
                radiation_gain=0.18,
                modal_mass_kg=0.024,
            ),
            Mode(
                1_480.0 if heavy else 2_200.0,
                0.048,
                input_gain=0.26,
                radiation_gain=0.13,
                modal_mass_kg=0.016,
            ),
            Mode(2_850.0, 0.030, input_gain=0.16, radiation_gain=0.09, modal_mass_kg=0.01),
        )
    )
    body = _modal_body(modes, radiation_gain=0.7)
    profile = FrictionProfile(
        base_gain=(0.13 if heavy else 0.09) * force_scale,
        roughness=0.66,
        noise_band_hz=(180.0, 3_600.0),
        stick_strength=0.33 * force_scale,
        slip_strength=0.22,
        f0_hz=(210.0, 620.0) if heavy else (360.0, 980.0),
        harmonic_rolloff=(1.25, 1.9),
        chaos_amount=0.26,
    )
    motion = _motion_curve(size=size, sample_rate=sample_rate)
    turn = render_sliding_source(
        motion=motion,
        profile=profile,
        sample_rate=sample_rate,
        seed=seed,
        event_index=4,
        body=body,
    )
    detent = render_terminal_closure(
        contact=ImpactContact(
            0.0028 if heavy else 0.0013,
            4.8e7 if heavy else 6.5e7,
            exponent=1.5,
            restitution=0.28,
        ),
        velocity_m_s=(0.38 if heavy else 0.27) * force_scale,
        modes=modes,
        sample_rate=sample_rate,
        seed=seed,
        event_index=5,
        output_gain=12.0,
        microscopic_gain=0.12,
        tail_s=0.20,
    )
    detent_start = round(0.78 * size)
    output = np.zeros(max(turn.size, detent_start + detent.size), dtype=np.float32)
    mix_at(output, turn, 0)
    mix_at(output, detent, detent_start)
    return output


def render_cloth_rustle(
    *, sample_rate: int, duration: float, fabric: str, activity_level: str, seed: int
) -> FloatAudio:
    size = max(1, round(duration * sample_rate))
    rng = component_rng(seed, _MATERIAL_NAMESPACE, 10, sample_rate)
    material_specs = {
        "cotton": ((420.0, 0.040, 0.40), (980.0, 0.028, 0.28), (2_100.0, 0.018, 0.16)),
        "silk": ((620.0, 0.028, 0.33), (1_480.0, 0.020, 0.26), (3_100.0, 0.014, 0.18)),
        "nylon": ((350.0, 0.034, 0.37), (1_100.0, 0.025, 0.29), (2_800.0, 0.016, 0.19)),
    }[fabric]
    modes = ModeSet(
        tuple(
            Mode(
                frequency * float(rng.uniform(0.97, 1.03)),
                decay,
                input_gain=gain,
                radiation_gain=gain,
                modal_mass_kg=0.015 + index * 0.008,
            )
            for index, (frequency, decay, gain) in enumerate(material_specs)
            if frequency < sample_rate * 0.43
        )
    )
    preset = ThinMaterialPreset(
        modes=modes,
        noise_band_hz=(240.0, min(7_200.0, sample_rate * 0.44)),
        event_rate_hz=22.0 if activity_level == "active" else 13.0,
        tail_s=0.08,
    )
    rate_hz = 2.0 if activity_level == "active" else 0.75
    events = sample_stochastic_events(
        duration_s=duration,
        sample_rate=sample_rate,
        rate_hz=rate_hz,
        seed=seed,
        component_id=11,
        event_duration_s=(0.18, 0.85),
    )
    activity = np.full(size, 0.045, dtype=np.float32)
    for event in events:
        end = min(size, event.start_sample + event.duration_samples)
        length = end - event.start_sample
        if length <= 0:
            continue
        position = np.linspace(0.0, 1.0, length, dtype=np.float32)
        envelope = np.maximum(
            np.sin(np.float32(math.pi) * position), np.float32(0.0)
        ) ** np.float32(1.2)
        activity[event.start_sample : end] = np.maximum(
            activity[event.start_sample : end], envelope * np.float32(event.level)
        )
    return render_thin_material_source(
        preset=preset,
        activity=activity,
        sample_rate=sample_rate,
        seed=seed,
        event_index=12,
        amplitude=0.18 if activity_level == "active" else 0.12,
    )


def render_floor_creak(
    *, sample_rate: int, duration: float, surface: str, weight: str, seed: int
) -> FloatAudio:
    size = max(1, round(duration * sample_rate))
    motion = _motion_curve(size=size, sample_rate=sample_rate)
    body = _floor_body(
        surface=surface, sample_rate=sample_rate, seed=seed, component_id=20, max_modes=26
    )
    load = 0.72 if weight == "heavy" else 0.42
    profile = FrictionProfile(
        base_gain=0.21 * load,
        roughness=0.72,
        noise_band_hz=(35.0, 1_500.0),
        stick_strength=0.48 * load,
        slip_strength=0.32 * load,
        f0_hz=(32.0, 96.0),
        harmonic_rolloff=(1.1, 1.8),
        chaos_amount=0.22,
    )
    result = render_sliding_source(
        motion=motion,
        profile=profile,
        sample_rate=sample_rate,
        seed=seed,
        event_index=21,
        body=body,
    )
    return result


def render_car_door(
    *, sample_rate: int, duration: float, action: str, size: str, force: str, seed: int
) -> FloatAudio:
    sample_count = max(1, round(duration * sample_rate))
    motion = _motion_curve(
        size=sample_count,
        sample_rate=sample_rate,
        reverse=action == "close",
    )
    body_gain = 1.0 if size == "sedan" else 1.18
    profile = FrictionProfile(
        base_gain=0.10 * body_gain,
        roughness=0.42,
        noise_band_hz=(55.0, 2_200.0),
        stick_strength=0.14,
        slip_strength=0.18,
        f0_hz=(65.0, 210.0),
        harmonic_rolloff=(1.5, 2.2),
        chaos_amount=0.18,
    )
    body_modes = ModeSet(
        (
            Mode(92.0, 0.15, input_gain=0.60, radiation_gain=0.26, modal_mass_kg=2.0),
            Mode(168.0, 0.12, input_gain=0.48, radiation_gain=0.23, modal_mass_kg=1.4),
            Mode(286.0, 0.085, input_gain=0.32, radiation_gain=0.18, modal_mass_kg=0.9),
            Mode(540.0, 0.060, input_gain=0.22, radiation_gain=0.13, modal_mass_kg=0.55),
            Mode(1_180.0, 0.035, input_gain=0.14, radiation_gain=0.09, modal_mass_kg=0.3),
        )
    )
    body = _modal_body(body_modes, radiation_gain=0.85 * body_gain, damping_scale=1.1)
    result = render_sliding_source(
        motion=motion,
        profile=profile,
        sample_rate=sample_rate,
        seed=seed,
        event_index=30,
        body=body,
    )
    contact = ImpactContact(
        0.48 if size == "sedan" else 0.72, 3.8e6, exponent=1.5, restitution=0.24
    )
    velocity = {"light": 0.34, "firm": 0.66}[force]
    terminal = render_terminal_closure(
        contact=contact,
        velocity_m_s=velocity,
        modes=body_modes,
        sample_rate=sample_rate,
        seed=seed,
        event_index=31,
        output_gain=17.0,
        microscopic_gain=0.12,
        tail_s=0.34,
    )
    terminal_start = sample_count - round(0.025 * sample_rate)
    output = np.zeros(
        max(result.size, sample_count, terminal_start + terminal.size), dtype=np.float32
    )
    mix_at(output, result, 0)
    if action == "close":
        mix_at(output, terminal, terminal_start)
    else:
        mix_at(output, terminal * np.float32(0.28), 0)
    return output

"""Semantic object interactions rendered through shared coupled impacts."""

from __future__ import annotations

import numpy as np

from ._dsp import mix_at
from ._physics.contact import ImpactContact, impact_force
from ._physics.geometry import rectangular_plate_modes
from ._physics.modes import Mode, ModeSet
from ._physics.physical_impacts import ImpactResponse, render_coupled_impact
from ._physics.presets import FLOOR_OBJECTS, OBJECT_PRESETS
from ._physics.rng import component_rng
from .types import FloatAudio

_OBJECT_NAMESPACE = 0x4F424A45


def _modes(
    seed: int,
    sample_rate: int,
    component_id: int,
    specifications: tuple[tuple[float, float, float], ...],
) -> ModeSet:
    rng = component_rng(seed, _OBJECT_NAMESPACE, component_id, sample_rate)
    return ModeSet(
        tuple(
            Mode(
                frequency * float(rng.uniform(0.985, 1.015)),
                decay * float(rng.uniform(0.92, 1.08)),
                input_gain=float(gain * rng.uniform(0.88, 1.12)),
                radiation_gain=gain,
                modal_mass_kg=0.02 + index * 0.01,
            )
            for index, (frequency, decay, gain) in enumerate(specifications)
            if frequency < sample_rate * 0.44
        )
    )


def _contact(
    seed: int,
    sample_rate: int,
    component_id: int,
    *,
    mass_kg: tuple[float, float],
    stiffness: tuple[float, float],
    restitution: float,
) -> ImpactContact:
    rng = component_rng(seed, _OBJECT_NAMESPACE, component_id, sample_rate)
    return ImpactContact(
        effective_mass_kg=float(rng.uniform(*mass_kg)),
        stiffness=float(rng.uniform(*stiffness)),
        exponent=1.5,
        restitution=restitution,
    )


def _impact(
    *,
    contact: ImpactContact,
    velocity_m_s: float,
    modes: tuple[tuple[ModeSet, float], ...],
    sample_rate: int,
    seed: int,
    event_index: int,
) -> FloatAudio:
    trace = impact_force(
        contact=contact,
        velocity_m_s=velocity_m_s,
        sample_rate=sample_rate,
    )
    responses = tuple(
        ImpactResponse(mode_set, gain=gain, microscopic_gain=0.10) for mode_set, gain in modes
    )
    rngs = tuple(
        component_rng(seed, _OBJECT_NAMESPACE, 80 + index, event_index)
        for index in range(len(responses))
    )
    return render_coupled_impact(trace, responses, sample_rate=sample_rate, rngs=rngs)


def _sequence(events: tuple[tuple[FloatAudio, int], ...]) -> FloatAudio:
    size = max(start + samples.size for samples, start in events)
    output = np.zeros(size, dtype=np.float32)
    for samples, start in events:
        mix_at(output, samples, start)
    return output


def render_switch_toggle(*, sample_rate: int, state: str, seed: int) -> FloatAudio:
    switch_modes = _modes(
        seed,
        sample_rate,
        1,
        ((420.0, 0.035, 0.42), (1_100.0, 0.026, 0.25), (2_300.0, 0.018, 0.13)),
    )
    frame_modes = _modes(
        seed,
        sample_rate,
        2,
        ((210.0, 0.075, 0.34), (380.0, 0.060, 0.28), (900.0, 0.038, 0.16), (1_800.0, 0.025, 0.09)),
    )
    contact = _contact(
        seed,
        sample_rate,
        3,
        mass_kg=(0.002, 0.005),
        stiffness=(2.0e7, 5.0e7),
        restitution=0.28,
    )
    initial_velocity, return_velocity = (0.44, 0.34) if state == "on" else (0.36, 0.43)
    responses = ((switch_modes, 0.65), (frame_modes, 0.85))
    first = _impact(
        contact=contact,
        velocity_m_s=initial_velocity,
        modes=responses,
        sample_rate=sample_rate,
        seed=seed,
        event_index=0,
    )
    release = _impact(
        contact=contact,
        velocity_m_s=return_velocity,
        modes=responses,
        sample_rate=sample_rate,
        seed=seed,
        event_index=1,
    )
    return _sequence(((first, 0), (release, round(0.075 * sample_rate))))


def render_button_press(*, sample_rate: int, size: str, force: str, seed: int) -> FloatAudio:
    button_specs = {
        "small": ((760.0, 0.035, 0.40), (1_650.0, 0.025, 0.28), (2_850.0, 0.018, 0.14)),
        "large": ((390.0, 0.060, 0.45), (920.0, 0.045, 0.30), (1_800.0, 0.032, 0.18)),
    }
    plunger_modes = _modes(seed, sample_rate, 10, button_specs[size])
    panel_modes = _modes(
        seed,
        sample_rate,
        11,
        ((260.0, 0.085, 0.34), (640.0, 0.060, 0.24), (1_400.0, 0.035, 0.12)),
    )
    mass, _stiffness = (0.0015, 0.004) if size == "small" else (0.004, 0.012)
    contact = _contact(
        seed,
        sample_rate,
        12,
        mass_kg=(mass, mass * 1.6),
        stiffness=(3.0e7, 9.0e7),
        restitution=0.24,
    )
    velocity = {"gentle": 0.20, "normal": 0.36, "firm": 0.56}[force]
    responses = ((plunger_modes, 0.62), (panel_modes, 0.78))
    press = _impact(
        contact=contact,
        velocity_m_s=velocity,
        modes=responses,
        sample_rate=sample_rate,
        seed=seed,
        event_index=0,
    )
    release = _impact(
        contact=contact,
        velocity_m_s=velocity * 0.55,
        modes=responses,
        sample_rate=sample_rate,
        seed=seed,
        event_index=1,
    )
    return _sequence(((press, 0), (release, round(0.065 * sample_rate))))


def _plate_modes(
    preset_name: str,
    *,
    seed: int,
    sample_rate: int,
    component_id: int,
    max_modes: int,
) -> ModeSet:
    preset = (
        OBJECT_PRESETS[preset_name] if preset_name in OBJECT_PRESETS else FLOOR_OBJECTS[preset_name]
    )
    rng = component_rng(seed, _OBJECT_NAMESPACE, component_id, sample_rate)
    return rectangular_plate_modes(
        preset.geometry,
        preset.material,
        sample_rate=sample_rate,
        max_modes=max_modes,
        contact_x=float(np.clip(0.5 + rng.normal(0.0, 0.06), 0.12, 0.88)),
        contact_y=float(np.clip(0.5 + rng.normal(0.0, 0.06), 0.12, 0.88)),
    )


def render_object_set_down(
    *, sample_rate: int, object_type: str, surface: str, force: str, seed: int
) -> FloatAudio:
    object_names = {"wood": "small_wood_board", "metal": "steel_sheet", "ceramic": "stone_tile"}
    object_modes = _plate_modes(
        object_names[object_type],
        seed=seed,
        sample_rate=sample_rate,
        component_id=20,
        max_modes=24,
    )
    surface_modes = _plate_modes(
        surface,
        seed=seed,
        sample_rate=sample_rate,
        component_id=21,
        max_modes=28,
    )
    contact_properties = {
        "wood": ((0.035, 0.070), (8.0e5, 2.2e6), 0.24),
        "metal": ((0.018, 0.042), (2.5e6, 6.0e6), 0.48),
        "ceramic": ((0.025, 0.055), (1.4e6, 4.0e6), 0.32),
    }[object_type]
    mass, stiffness, restitution = contact_properties
    contact = _contact(
        seed,
        sample_rate,
        22,
        mass_kg=mass,
        stiffness=stiffness,
        restitution=restitution,
    )
    velocity = {"gentle": 0.13, "normal": 0.28, "firm": 0.48}[force]
    return _impact(
        contact=contact,
        velocity_m_s=velocity,
        modes=((object_modes, 0.82), (surface_modes, 0.68)),
        sample_rate=sample_rate,
        seed=seed,
        event_index=0,
    )


def render_glass_clink(
    *, sample_rate: int, style: str, count: int, force: str, seed: int
) -> FloatAudio:
    glass_specs = {
        "wine": (
            (430.0, 0.42, 0.72),
            (860.0, 0.34, 0.48),
            (1_420.0, 0.25, 0.30),
            (2_250.0, 0.19, 0.18),
            (3_250.0, 0.14, 0.10),
        ),
        "tumbler": (
            (560.0, 0.28, 0.72),
            (1_120.0, 0.23, 0.48),
            (1_740.0, 0.18, 0.30),
            (2_480.0, 0.14, 0.18),
            (3_400.0, 0.11, 0.10),
        ),
    }
    first = _modes(seed, sample_rate, 30, glass_specs[style])
    second_specs = tuple(
        (frequency * 1.035, decay * 0.92, gain * 0.82)
        for frequency, decay, gain in glass_specs[style]
    )
    second = _modes(seed, sample_rate, 31, second_specs)
    contact = _contact(
        seed,
        sample_rate,
        32,
        mass_kg=(0.008, 0.018),
        stiffness=(1.5e7, 4.5e7),
        restitution=0.60,
    )
    velocity = {"gentle": 0.14, "normal": 0.23, "firm": 0.34}[force]
    responses = ((first, 0.74), (second, 0.60))
    events = tuple(
        (
            _impact(
                contact=contact,
                velocity_m_s=velocity * float(1.0 + 0.035 * (index % 2)),
                modes=responses,
                sample_rate=sample_rate,
                seed=seed,
                event_index=index,
            ),
            round(index * 0.22 * sample_rate),
        )
        for index in range(count)
    )
    return _sequence(events)

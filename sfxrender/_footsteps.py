"""GRF-driven footstep synthesis for solid and aggregate materials."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from functools import lru_cache

import numpy as np

from ._dsp import (
    asymmetric_pulse,
    bandpass_noise,
    mix_at,
    one_pole_highpass,
    one_pole_lowpass,
)
from ._foley_profiles import (
    AGGREGATE_SURFACE_PROFILES,
    FOOTWEAR_PROFILES,
    SOLID_SURFACE_PROFILES,
    AggregateSurfaceProfile,
    FootwearProfile,
)
from ._physics.contact import ImpactContact, impact_force
from ._physics.friction import (
    LuGrePreset,
    lugre_friction,
    make_roughness_profile,
    roughness_velocity,
)
from ._physics.geometry import BoundaryCondition, RectangularPlate, rectangular_plate_modes
from ._physics.modes import ModeSet
from ._physics.physical_impacts import render_physical_impact
from ._physics.presets import FLOOR_OBJECTS, FOOTWEAR_CONTACTS, FootwearContactPreset
from ._physics.resonator import ModalResonatorBank
from .types import FloatAudio


@dataclass(frozen=True, slots=True)
class FootstepImpact:
    time_s: float
    amplitude: float
    role: str
    hardness: float
    contact_x: float
    contact_y: float


@dataclass(frozen=True, slots=True)
class FrictionEvent:
    start_s: float
    duration_s: float
    gain: float
    velocity_scale: float
    role: str


@dataclass(frozen=True, slots=True)
class FootstepExciter:
    envelope: FloatAudio
    impacts: tuple[FootstepImpact, ...]
    friction_events: tuple[FrictionEvent, ...]
    load_scale: float


def _normalized_pace(gait: str, step_interval_s: float) -> float:
    reference_interval = {
        "walk": 0.72,
        "run": 0.40,
        "stairs_up": 0.64,
        "stairs_down": 0.64,
    }[gait]
    return float(
        np.clip((reference_interval - step_interval_s) / (reference_interval * 0.62), 0.0, 1.0)
    )


def generate_footstep_exciter(
    *,
    sample_rate: int,
    footwear: str,
    force: float,
    gait: str,
    step_interval_s: float,
    rng: np.random.Generator,
) -> FootstepExciter:
    """Build a pace-sensitive foot-ground event grammar and control envelope."""
    profile = FOOTWEAR_PROFILES[footwear]
    pace = _normalized_pace(gait, step_interval_s)
    duration_scale = 1.10 + (0.72 - 1.10) * pace
    heel_toe_scale = 1.10 + (0.55 - 1.10) * pace
    pace_load_scale = 0.92 + (1.22 - 0.92) * pace
    force_level = float(np.clip((force - 0.05) / 0.95, 0.0, 1.0))
    load_scale = pace_load_scale
    contact_width = 1.0 + 0.12 * force_level
    sole_delay = float(rng.uniform(*profile.sole_delay_s)) * heel_toe_scale
    sole_duration = float(rng.uniform(*profile.sole_duration_s)) * duration_scale
    toe_delay = float(rng.uniform(*profile.toe_delay_s)) * heel_toe_scale
    toe_duration = float(rng.uniform(*profile.toe_duration_s)) * duration_scale
    if gait == "run":
        sole_delay *= 0.78
        sole_duration *= 0.82
        toe_delay *= 0.76
        toe_duration *= 0.82
    elif gait == "stairs_up":
        toe_delay *= 0.78
        toe_duration *= 0.90
    elif gait == "stairs_down":
        toe_delay *= 1.10

    peak_load = (0.18 + 1.35 * force_level**1.3) * load_scale * float(rng.uniform(0.97, 1.03))
    heel_amplitude = peak_load * (0.30 + 0.20 * profile.heel_hardness)
    sole_amplitude = peak_load * (0.52 + 0.24 * profile.low_weight_gain)
    toe_amplitude = peak_load * (0.16 + 0.12 * profile.toe_gain)
    if gait == "run":
        heel_amplitude *= 1.08
        sole_amplitude *= 1.05
    elif gait == "stairs_up":
        heel_amplitude *= 0.72
        toe_amplitude *= 1.30
    elif gait == "stairs_down":
        heel_amplitude *= 1.30
        toe_amplitude *= 0.74

    heel_attack = profile.heel_attack_s * duration_scale
    heel_decay = profile.heel_decay_s * duration_scale * contact_width
    size_seconds = (
        max(
            0.22,
            sole_delay + sole_duration * 1.15,
            toe_delay + toe_duration * 1.15,
            heel_attack + heel_decay,
        )
        + 0.03
    )
    size = max(1, round(size_seconds * sample_rate))
    envelope = np.zeros(size, dtype=np.float32)
    components: list[tuple[str, float, float, float, float]] = [
        ("heel", 0.0, heel_attack, heel_decay, heel_amplitude),
        (
            "sole",
            sole_delay,
            sole_duration * float(rng.uniform(0.16, 0.34)),
            sole_duration * contact_width,
            sole_amplitude * float(rng.uniform(0.88, 1.12)),
        ),
    ]
    toe_probability = {"walk": 0.94, "run": 0.78, "stairs_up": 0.96, "stairs_down": 0.84}[gait]
    if rng.random() < toe_probability:
        components.append(
            (
                "toe",
                toe_delay,
                toe_duration * float(rng.uniform(0.10, 0.28)),
                toe_duration * float(rng.uniform(0.62, 1.08)),
                toe_amplitude * float(rng.uniform(0.72, 1.22)),
            )
        )
    impacts: list[FootstepImpact] = []
    for role, onset, attack, decay, amplitude in components:
        envelope += asymmetric_pulse(
            size,
            sample_rate,
            onset_s=onset,
            attack_s=attack,
            decay_s=decay,
            amplitude=amplitude,
        )
        marker_time = onset + max(1.0 / sample_rate, attack)
        contact_x = float(rng.uniform(0.12, 0.35) if role == "heel" else rng.uniform(0.58, 0.88))
        if gait == "stairs_up":
            contact_y = float(rng.uniform(0.65, 0.92))
        elif gait == "stairs_down":
            contact_y = float(rng.uniform(0.08, 0.36))
        else:
            contact_y = float(rng.uniform(0.30, 0.72))
        hardness = profile.heel_hardness if role == "heel" else profile.heel_hardness * 0.82
        impacts.append(FootstepImpact(marker_time, amplitude, role, hardness, contact_x, contact_y))

    friction_events: list[FrictionEvent] = []
    scuff_probability = {
        "walk": 0.14 + 0.16 * pace,
        "run": 0.32 + 0.28 * pace,
        "stairs_up": 0.14 + 0.12 * pace,
        "stairs_down": 0.20 + 0.16 * pace,
    }[gait]
    scuff_probability *= 0.55 + 0.45 * min(1.0, profile.friction_gain / 0.5)
    toe_start = next((impact.time_s for impact in impacts if impact.role == "toe"), sole_delay)
    if rng.random() < scuff_probability:
        friction_events.append(
            FrictionEvent(
                start_s=max(0.0, toe_start - float(rng.uniform(0.004, 0.018))),
                duration_s=float(rng.uniform(0.018, 0.052)) * duration_scale,
                gain=profile.friction_gain * (0.16 + 0.22 * pace) * load_scale,
                velocity_scale=float(rng.uniform(0.75, 1.35)) * (0.85 + 0.3 * pace),
                role="scuff",
            )
        )
    brush_probability = 0.10 if footwear in {"barefoot", "boots"} else 0.055
    if gait.startswith("stairs_"):
        brush_probability *= 0.65
    if rng.random() < brush_probability:
        friction_events.append(
            FrictionEvent(
                start_s=max(0.0, sole_delay - float(rng.uniform(0.012, 0.030))),
                duration_s=float(rng.uniform(0.012, 0.032)) * duration_scale,
                gain=profile.friction_gain * 0.12 * load_scale,
                velocity_scale=float(rng.uniform(0.55, 0.95)),
                role="brush",
            )
        )

    return FootstepExciter(
        envelope=envelope,
        impacts=tuple(impacts),
        friction_events=tuple(friction_events),
        load_scale=load_scale,
    )


def _synthetic_grf(
    *,
    sample_rate: int,
    footwear: str,
    force: float,
    rng: np.random.Generator,
    gait: str = "walk",
    step_interval_s: float = 0.52,
) -> FloatAudio:
    """Compatibility wrapper returning the exciter's ground-load envelope."""
    return generate_footstep_exciter(
        sample_rate=sample_rate,
        footwear=footwear,
        force=force,
        gait=gait,
        step_interval_s=step_interval_s,
        rng=rng,
    ).envelope


@lru_cache(maxsize=64)
def _floor_modes(
    surface_name: str,
    sample_rate: int,
    contact_x: float,
    contact_y: float,
) -> ModeSet:
    """Build stable geometry-derived floor modes at a normalized contact point."""
    floor = FLOOR_OBJECTS[surface_name]
    base = rectangular_plate_modes(
        floor.geometry,
        floor.material,
        sample_rate=sample_rate,
        max_modes=40,
        contact_x=contact_x,
        contact_y=contact_y,
    )
    return ModeSet(
        tuple(
            replace(
                mode,
                decay_s=mode.decay_s * floor.mode_decay_scale,
                input_gain=mode.input_gain * floor.modal_gain,
                modal_mass_kg=mode.modal_mass_kg * floor.modal_mass_scale,
            )
            for mode in base.modes
        )
    )


@lru_cache(maxsize=64)
def _stair_modes(
    surface_name: str,
    sample_rate: int,
    contact_x: float,
    contact_y: float,
) -> ModeSet:
    """Build a smaller tread response from the selected stair material."""
    floor = FLOOR_OBJECTS[surface_name]
    thickness = 0.032 if surface_name == "wood" else 0.045
    tread = RectangularPlate(1.0, 0.28, thickness, BoundaryCondition.FLOOR_SUPPORTED)
    base = rectangular_plate_modes(
        tread,
        floor.material,
        sample_rate=sample_rate,
        max_modes=40,
        contact_x=contact_x,
        contact_y=contact_y,
    )
    return ModeSet(
        tuple(
            replace(
                mode,
                decay_s=mode.decay_s * floor.mode_decay_scale,
                input_gain=mode.input_gain * floor.modal_gain,
                modal_mass_kg=mode.modal_mass_kg * floor.modal_mass_scale,
            )
            for mode in base.modes
        )
    )


def _interaction_modes(
    surface: str, gait: str, sample_rate: int, contact_x: float, contact_y: float
) -> ModeSet:
    x = round(float(np.clip(contact_x, 0.08, 0.92)) * 8.0) / 8.0
    y = round(float(np.clip(contact_y, 0.08, 0.92)) * 8.0) / 8.0
    if gait.startswith("stairs_"):
        return _stair_modes(surface, sample_rate, x, y)
    return _floor_modes(surface, sample_rate, x, y)


def _floor_contact_audio(
    *,
    exciter: FootstepExciter,
    surface: str,
    footwear: FootwearProfile,
    footwear_contact: FootwearContactPreset,
    rng: np.random.Generator,
    sample_rate: int,
    gait: str = "walk",
) -> tuple[FloatAudio, ModeSet]:
    envelope = exciter.envelope
    size = envelope.size
    floor = FLOOR_OBJECTS[surface]
    if size == 0:
        return np.zeros(0, dtype=np.float32), _interaction_modes(
            surface, gait, sample_rate, 0.5, 0.5
        )

    event_modes = tuple(
        _interaction_modes(surface, gait, sample_rate, impact.contact_x, impact.contact_y)
        for impact in exciter.impacts
    )
    center_modes = _interaction_modes(surface, gait, sample_rate, 0.5, 0.5)
    max_decay = max(
        (mode.decay_s for mode_set in event_modes for mode in mode_set.modes),
        default=0.04,
    )
    tail_samples = round(min(0.55, max(0.12, max_decay * 4.5)) * sample_rate)
    output = np.zeros(size + tail_samples, dtype=np.float32)
    role_gain = {"heel": 1.0, "sole": 0.82, "toe": 0.62}

    for impact, modes in zip(exciter.impacts, event_modes, strict=True):
        event_start = round(impact.time_s * sample_rate)
        if event_start >= size:
            continue
        hardness = float(np.clip(impact.hardness, 0.0, 1.0))
        stiffness = footwear_contact.normal_stiffness_n_m * (0.72 + 0.56 * hardness)
        restitution = min(
            1.0,
            footwear_contact.material.restitution * (0.84 + 0.30 * hardness),
        )
        contact = ImpactContact(
            effective_mass_kg=footwear_contact.effective_mass_kg,
            stiffness=stiffness,
            restitution=restitution,
        )
        velocity = 0.045 + 0.22 * math.sqrt(max(impact.amplitude, 0.0)) * (0.65 + 0.35 * hardness)
        trace = impact_force(
            contact=contact,
            velocity_m_s=velocity,
            sample_rate=sample_rate,
            max_contact_s=0.06,
        )
        event_audio = render_physical_impact(
            trace,
            modes,
            sample_rate=sample_rate,
            rng=rng,
            microscopic_gain=0.08 + 0.55 * hardness,
            output_gain=(
                22.0
                * floor.modal_gain
                * SOLID_SURFACE_PROFILES[surface].transient_gain
                * role_gain.get(impact.role, 0.7)
                * (0.72 + 0.28 * footwear.heel_gain)
            ),
        )
        mix_at(output, event_audio, event_start)
    return output, center_modes


def _mechanical_friction_audio(
    *,
    grf: FloatAudio,
    friction_events: tuple[FrictionEvent, ...],
    footwear: FootwearProfile,
    footwear_contact: FootwearContactPreset,
    surface: str,
    modes: ModeSet,
    rng: np.random.Generator,
    sample_rate: int,
) -> FloatAudio:
    output = np.zeros(grf.size, dtype=np.float32)
    if grf.size == 0 or not friction_events:
        return output
    floor_material = FLOOR_OBJECTS[surface].material
    friction_scale = 0.48 + 0.52 * footwear.friction_gain
    mu_static = (
        math.sqrt(footwear_contact.material.friction_static * floor_material.friction_static)
        * friction_scale
    )
    mu_dynamic = (
        math.sqrt(footwear_contact.material.friction_dynamic * floor_material.friction_dynamic)
        * friction_scale
    )
    preset = LuGrePreset(
        static_coefficient=mu_static,
        dynamic_coefficient=min(mu_static, mu_dynamic),
        stribeck_velocity_m_s=max(
            0.005,
            math.sqrt(
                footwear_contact.material.stribeck_velocity_m_s
                * floor_material.stribeck_velocity_m_s
            ),
        ),
        bristle_stiffness_n_m=max(4_000.0, footwear_contact.normal_stiffness_n_m * 0.045),
        bristle_damping_n_s_m=0.12,
        viscous_coefficient_n_s_m=(
            math.sqrt(footwear_contact.material.viscous_friction * floor_material.viscous_friction)
            * 0.08
        ),
    )
    roughness_rms = math.sqrt(
        footwear_contact.material.roughness_rms_m**2 + floor_material.roughness_rms_m**2
    )
    correlation = math.sqrt(
        footwear_contact.material.roughness_correlation_m * floor_material.roughness_correlation_m
    )
    for event in friction_events:
        start = round(event.start_s * sample_rate)
        if start >= grf.size:
            continue
        event_size = min(round(event.duration_s * sample_rate), grf.size - start)
        if event_size <= 1:
            continue
        load = grf[start : start + event_size]
        phase = np.linspace(0.0, 1.0, event_size, dtype=np.float32)
        window = np.maximum(0.0, np.sin(np.float32(math.pi) * phase)) ** np.float32(0.7)
        slip_velocity = window * (np.float32(0.035) + np.float32(0.18) * load)
        slip_velocity *= np.float32(event.velocity_scale)
        normal_load = load * np.float32(620.0)
        trace = lugre_friction(
            np.asarray(slip_velocity, dtype=np.float32),
            np.asarray(normal_load, dtype=np.float32),
            preset,
            sample_rate,
        )
        traveled = np.cumsum(np.abs(slip_velocity), dtype=np.float64) / sample_rate
        roughness = make_roughness_profile(
            roughness_rms_m=max(roughness_rms, 1e-9),
            correlation_length_m=max(correlation, 1e-6),
            seed=int(rng.integers(0, 2**32, dtype=np.uint32)),
            length_m=max(0.12, float(traveled[-1]) + 0.05),
        )
        surface_velocity = roughness_velocity(
            roughness, np.asarray(traveled, dtype=np.float32), sample_rate
        )
        power_envelope = np.sqrt(trace.power_w / np.float32(0.8))
        release = np.sqrt(trace.release_energy_j * np.float32(sample_rate)) * np.float32(0.008)
        excitation = np.asarray(
            surface_velocity * power_envelope * np.float32(150.0) + release,
            dtype=np.float32,
        )
        resonant = ModalResonatorBank(modes, sample_rate).process(excitation)
        direct = excitation * np.float32(
            0.025 * footwear.friction_gain * SOLID_SURFACE_PROFILES[surface].friction_roughness
        )
        result = np.asarray(direct + resonant * np.float32(90.0), dtype=np.float32)
        result *= np.float32(event.gain * 4.0)
        if footwear.friction_brightness < 0.5:
            result = one_pole_lowpass(result, sample_rate, 2_200.0)
        else:
            result = one_pole_highpass(result, sample_rate, 480.0)
        mix_at(output, result, start)
    return output


def _solid_footstep(
    *,
    exciter: FootstepExciter,
    surface: str,
    footwear: FootwearProfile,
    footwear_contact: FootwearContactPreset,
    rng: np.random.Generator,
    sample_rate: int,
    gait: str = "walk",
) -> FloatAudio:
    """Render exciter-driven contacts, impact-local texture, and event friction."""
    profile = SOLID_SURFACE_PROFILES[surface]
    grf = exciter.envelope
    size = grf.size
    if size == 0:
        return np.zeros(0, dtype=np.float32)
    physical_contact, floor_modes = _floor_contact_audio(
        exciter=exciter,
        surface=surface,
        footwear=footwear,
        footwear_contact=footwear_contact,
        rng=rng,
        sample_rate=sample_rate,
        gait=gait,
    )
    low_noise = bandpass_noise(rng, size, sample_rate, 42.0, 850.0)
    mid_noise = bandpass_noise(
        rng,
        size,
        sample_rate,
        max(100.0, profile.low_cut_hz),
        min(4_200.0, sample_rate * 0.44),
    )
    low_layer = low_noise * grf * np.float32(0.31 * footwear.low_weight_gain)
    mid_layer = mid_noise * grf * np.float32(0.41 * footwear.sole_gain)
    attack_noise = np.zeros(size, dtype=np.float32)
    for impact in exciter.impacts:
        start = round(impact.time_s * sample_rate)
        if start >= size:
            continue
        burst_size = min(max(2, round(0.010 * sample_rate)), size - start)
        high_hz = min(profile.high_cut_hz, sample_rate * 0.46)
        low_hz = min(max(900.0, profile.low_cut_hz), high_hz - 1.0)
        noise = bandpass_noise(rng, burst_size, sample_rate, low_hz, high_hz)
        time = np.arange(burst_size, dtype=np.float32) / np.float32(sample_rate)
        attack = max(1.0 / sample_rate, footwear.heel_attack_s * (1.0 - 0.35 * impact.hardness))
        decay = max(0.0015, 0.0065 * (1.0 - 0.42 * impact.hardness))
        envelope = (1.0 - np.exp(-time / np.float32(attack))) * np.exp(-time / np.float32(decay))
        gain = impact.amplitude * (0.08 + 0.22 * impact.hardness) * profile.transient_gain
        mix_at(
            attack_noise,
            np.asarray(noise * envelope * np.float32(gain), dtype=np.float32),
            start,
        )
    contact_noise = np.asarray(
        physical_contact[:size] * np.float32(0.08) + low_layer + mid_layer + attack_noise,
        dtype=np.float32,
    )
    contact_noise = one_pole_lowpass(contact_noise, sample_rate, profile.high_cut_hz)
    contact_noise = one_pole_highpass(contact_noise, sample_rate, profile.low_cut_hz)
    output = physical_contact.copy()
    output[:size] += contact_noise
    friction = _mechanical_friction_audio(
        grf=grf,
        friction_events=exciter.friction_events,
        footwear=footwear,
        footwear_contact=footwear_contact,
        surface=surface,
        modes=floor_modes,
        rng=rng,
        sample_rate=sample_rate,
    )
    output[:size] += friction
    if surface == "wood":
        output = one_pole_lowpass(output, sample_rate, min(3_200.0, sample_rate * 0.42))
    elif surface == "carpet":
        output = one_pole_lowpass(output, sample_rate, profile.high_cut_hz)
    return np.asarray(output, dtype=np.float32)


def solid_footstep(
    *,
    sample_rate: int,
    surface: str,
    footwear: str,
    force: float,
    rng: np.random.Generator,
    gait: str = "walk",
    step_interval_s: float = 0.52,
) -> FloatAudio:
    """Synthesize one exciter-driven solid-surface step."""
    exciter = generate_footstep_exciter(
        sample_rate=sample_rate,
        footwear=footwear,
        force=force,
        gait=gait,
        step_interval_s=step_interval_s,
        rng=rng,
    )
    return _solid_footstep(
        exciter=exciter,
        surface=surface,
        footwear=FOOTWEAR_PROFILES[footwear],
        footwear_contact=FOOTWEAR_CONTACTS[footwear],
        rng=rng,
        sample_rate=sample_rate,
        gait=gait,
    )


@dataclass(frozen=True, slots=True)
class ParticleEvent:
    """One sampled aggregate micro-collision."""

    start_sample: int
    duration_samples: int
    energy: float
    center_hz: float
    resonant: bool
    population: str

    work_budget_j: float


def _sample_particle_events(
    *,
    exciter: FootstepExciter,
    surface: AggregateSurfaceProfile,
    rng: np.random.Generator,
    sample_rate: int,
    footwear_contact: FootwearContactPreset | None = None,
    friction_gain: float = 0.5,
) -> list[ParticleEvent]:
    """Sample three load-driven grain families under an exciter work budget."""
    load_envelope = exciter.envelope
    block_size = max(1, round(0.002 * sample_rate))
    low_hz, high_hz = surface.particle_band_hz
    high_hz = min(high_hz, sample_rate * 0.43)
    low_hz = min(low_hz, high_hz * 0.45)
    populations = (
        ("small", max(low_hz, high_hz * 0.52), high_hz, 0.70, 1.25),
        ("medium", max(low_hz, high_hz * 0.20), min(high_hz, high_hz * 0.58), 1.0, 1.0),
        ("large", low_hz, min(high_hz, high_hz * 0.28), 1.45, 0.72),
    )
    events: list[ParticleEvent] = []
    material_mu = footwear_contact.material.friction_dynamic if footwear_contact else 0.78
    footwear_scale = 0.48 + 0.52 * friction_gain
    for block_start in range(0, load_envelope.size, block_size):
        block_end = min(load_envelope.size, block_start + block_size)
        block = load_envelope[block_start:block_end]
        mean_load = max(
            0.0,
            float(np.mean(block, dtype=np.float64)) * exciter.load_scale,
        )
        rate = surface.event_density * mean_load**surface.density_force_exponent
        count = int(rng.poisson(rate * (block_end - block_start) / sample_rate))
        block_duration = (block_end - block_start) / sample_rate
        slip_work_j = (
            mean_load * 620.0 * material_mu * footwear_scale * 0.25 * block_duration * 0.42
        )
        if count == 0 or slip_work_j < surface.minimum_energy:
            continue
        max_events = max(1, int(slip_work_j / surface.minimum_energy))
        count = min(count, max_events)
        remaining_work_j = slip_work_j
        for _ in range(count):
            draw = float(rng.random())
            if draw < 0.55:
                population, band_low, band_high, duration_scale, energy_exponent = populations[0]
            elif draw < 0.87:
                population, band_low, band_high, duration_scale, energy_exponent = populations[1]
            else:
                population, band_low, band_high, duration_scale, energy_exponent = populations[2]
            duration = max(
                2,
                round(
                    float(rng.uniform(*surface.particle_duration_s)) * duration_scale * sample_rate
                ),
            )
            u = float(rng.random())
            proposed_energy = surface.minimum_energy + (
                surface.maximum_energy - surface.minimum_energy
            ) * u ** (surface.energy_power * energy_exponent)
            energy = min(proposed_energy, remaining_work_j)
            if energy < surface.minimum_energy:
                break
            remaining_work_j -= energy
            center = math.exp(float(rng.uniform(math.log(band_low), math.log(band_high))))
            events.append(
                ParticleEvent(
                    start_sample=int(rng.integers(block_start, block_end)),
                    duration_samples=duration,
                    energy=energy,
                    center_hz=center,
                    resonant=bool(rng.random() < surface.resonance_probability),
                    work_budget_j=slip_work_j,
                    population=population,
                )
            )
    return events


def _aggregate_footstep(
    *,
    exciter: FootstepExciter,
    surface: AggregateSurfaceProfile,
    footwear: FootwearProfile,
    footwear_contact: FootwearContactPreset,
    rng: np.random.Generator,
    sample_rate: int,
    gait: str = "walk",
) -> FloatAudio:
    """Render load-bounded granular contacts over a physical hard-floor response."""
    grf = exciter.envelope
    if grf.size == 0:
        return np.zeros(0, dtype=np.float32)
    particle_tail = max(1, round(surface.particle_duration_s[1] * 1.45 * sample_rate))
    body, floor_modes = _floor_contact_audio(
        exciter=exciter,
        surface="stone",
        footwear=footwear,
        footwear_contact=footwear_contact,
        rng=rng,
        sample_rate=sample_rate,
        gait=gait,
    )
    output = np.zeros(max(body.size, grf.size + particle_tail), dtype=np.float32)
    output[: body.size] += body
    body_noise = bandpass_noise(rng, grf.size, sample_rate, 45.0, 1_400.0)
    output[: grf.size] += body_noise * grf * np.float32(0.13 * footwear.low_weight_gain)
    events = _sample_particle_events(
        exciter=exciter,
        surface=surface,
        rng=rng,
        sample_rate=sample_rate,
        footwear_contact=footwear_contact,
        friction_gain=footwear.friction_gain,
    )
    for event in events:
        size = event.duration_samples
        noise_low = max(80.0, event.center_hz / 1.65)
        noise_high = min(event.center_hz * 1.65, sample_rate * 0.47)
        noise = bandpass_noise(rng, size, sample_rate, noise_low, noise_high)
        t = np.arange(size, dtype=np.float32) / np.float32(sample_rate)
        attack = float(rng.uniform(0.00012, 0.00065))
        decay = float(rng.uniform(0.0012, 0.0065))
        envelope = (1.0 - np.exp(-t / np.float32(attack))) * np.exp(-t / np.float32(decay))
        peak = float(np.max(envelope)) if envelope.size else 0.0
        if peak > 0.0:
            envelope /= np.float32(peak)
        amplitude = math.sqrt(event.energy) * surface.texture_gain
        particle = noise * envelope * np.float32(amplitude)
        if event.resonant:
            phase = float(rng.uniform(-math.pi, math.pi))
            ring = np.sin(
                np.float32(2.0 * math.pi * event.center_hz) * t + np.float32(phase)
            ) * np.exp(-t / np.float32(max(0.001, decay * 0.7)))
            particle += ring * envelope * np.float32(amplitude * 0.075)
        mix_at(output, np.asarray(particle, dtype=np.float32), event.start_sample)
    friction = _mechanical_friction_audio(
        grf=grf,
        friction_events=exciter.friction_events,
        footwear=footwear,
        footwear_contact=footwear_contact,
        surface="stone",
        modes=floor_modes,
        rng=rng,
        sample_rate=sample_rate,
    )
    output[: grf.size] += friction
    return output


def aggregate_footstep(
    *,
    sample_rate: int,
    surface: str,
    footwear: str,
    force: float,
    rng: np.random.Generator,
    gait: str = "walk",
    step_interval_s: float = 0.52,
) -> FloatAudio:
    """Synthesize one seeded exciter-driven aggregate-surface step."""
    exciter = generate_footstep_exciter(
        sample_rate=sample_rate,
        footwear=footwear,
        force=force,
        gait=gait,
        step_interval_s=step_interval_s,
        rng=rng,
    )
    return _aggregate_footstep(
        exciter=exciter,
        surface=AGGREGATE_SURFACE_PROFILES[surface],
        footwear=FOOTWEAR_PROFILES[footwear],
        footwear_contact=FOOTWEAR_CONTACTS[footwear],
        rng=rng,
        sample_rate=sample_rate,
        gait=gait,
    )

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
from ._physics.geometry import rectangular_plate_modes
from ._physics.modes import ModeSet
from ._physics.physical_impacts import render_physical_impact
from ._physics.presets import FLOOR_OBJECTS, FOOTWEAR_CONTACTS, FootwearContactPreset
from ._physics.resonator import ModalResonatorBank
from .types import FloatAudio


def _synthetic_grf(
    *,
    sample_rate: int,
    footwear: str,
    force: float,
    rng: np.random.Generator,
) -> FloatAudio:
    """Return a seeded heel/load/toe ground-reaction-force control envelope."""
    profile = FOOTWEAR_PROFILES[footwear]
    sole_delay = float(rng.uniform(*profile.sole_delay_s))
    sole_duration = float(rng.uniform(*profile.sole_duration_s))
    toe_delay = float(rng.uniform(*profile.toe_delay_s))
    toe_duration = float(rng.uniform(*profile.toe_duration_s))
    size_seconds = max(0.22, toe_delay + toe_duration + 0.032)
    size = max(1, round(size_seconds * sample_rate))

    force_level = float(np.clip((force - 0.05) / 0.95, 0.0, 1.0))
    peak_load = 0.28 + 0.95 * force_level
    peak_load *= float(rng.uniform(0.97, 1.03))
    contact_width = 1.0 + 0.12 * force_level
    heel_amplitude = peak_load * (0.30 + 0.20 * profile.heel_hardness)
    sole_amplitude = peak_load * (0.52 + 0.24 * profile.low_weight_gain)
    toe_amplitude = peak_load * (0.16 + 0.12 * profile.toe_gain)

    heel = asymmetric_pulse(
        size,
        sample_rate,
        attack_s=profile.heel_attack_s,
        decay_s=profile.heel_decay_s * contact_width,
        amplitude=heel_amplitude * float(rng.uniform(0.90, 1.10)),
    )
    sole = asymmetric_pulse(
        size,
        sample_rate,
        onset_s=sole_delay,
        attack_s=sole_duration * float(rng.uniform(0.16, 0.34)),
        decay_s=sole_duration * contact_width,
        amplitude=sole_amplitude * float(rng.uniform(0.88, 1.12)),
    )
    toe = asymmetric_pulse(
        size,
        sample_rate,
        onset_s=toe_delay,
        attack_s=toe_duration * float(rng.uniform(0.10, 0.28)),
        decay_s=toe_duration * float(rng.uniform(0.62, 1.08)),
        amplitude=toe_amplitude * float(rng.uniform(0.72, 1.22)),
    )
    grf = np.asarray(heel + sole + toe, dtype=np.float32)
    maximum = float(np.max(grf)) if grf.size else 0.0
    if maximum > 1.0:
        grf /= np.float32(maximum)
    return grf


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


def _floor_contact_audio(
    *,
    grf: FloatAudio,
    surface: str,
    footwear: FootwearProfile,
    footwear_contact: FootwearContactPreset,
    rng: np.random.Generator,
    sample_rate: int,
) -> tuple[FloatAudio, ModeSet]:
    size = grf.size
    floor = FLOOR_OBJECTS[surface]
    contacts = (
        (0.0, 0.042, 1.0),
        (0.032, 0.135, 0.76),
        (0.105, 0.225, 0.54),
    )
    mode_sets = (
        _floor_modes(surface, sample_rate, 0.20, 0.40),
        _floor_modes(surface, sample_rate, 0.62, 0.56),
        _floor_modes(surface, sample_rate, 0.82, 0.68),
    )
    max_decay = max(
        (mode.decay_s for mode_set in mode_sets for mode in mode_set.modes),
        default=0.04,
    )
    tail_samples = round(min(0.55, max(0.12, max_decay * 4.5)) * sample_rate)
    output = np.zeros(size + tail_samples, dtype=np.float32)
    for stage_index, (start_s, end_s, stage_gain) in enumerate(contacts):
        start_index = min(size, round(start_s * sample_rate))
        end_index = min(size, round(end_s * sample_rate))
        stage = grf[start_index:end_index]
        peak = float(np.max(stage)) if stage.size else 0.0
        if peak <= 1e-8:
            continue
        active = np.flatnonzero(stage >= peak * 0.24)
        event_start = start_index + (int(active[0]) if active.size else 0)
        effective_mass = footwear_contact.effective_mass_kg
        stiffness = footwear_contact.normal_stiffness_n_m * (1.0 - 0.14 * stage_index)
        contact = ImpactContact(
            effective_mass_kg=effective_mass,
            stiffness=stiffness,
            restitution=footwear_contact.material.restitution,
        )
        velocity = 0.045 + 0.22 * math.sqrt(peak) * (
            0.65 + 0.35 * footwear_contact.material.contact_hardness
        )
        trace = impact_force(
            contact=contact,
            velocity_m_s=velocity,
            sample_rate=sample_rate,
            max_contact_s=0.06,
        )
        event = render_physical_impact(
            trace,
            mode_sets[stage_index],
            sample_rate=sample_rate,
            rng=rng,
            microscopic_gain=(
                0.08
                if surface == "wood"
                else 0.25 + 0.62 * footwear_contact.material.contact_hardness
            ),
            output_gain=(
                22.0
                * floor.modal_gain
                * SOLID_SURFACE_PROFILES[surface].transient_gain
                * stage_gain
                * (0.72 + 0.28 * footwear.heel_gain)
            ),
        )
        mix_at(output, event, event_start)
    return output, mode_sets[1]


def _mechanical_friction_audio(
    *,
    grf: FloatAudio,
    footwear: FootwearProfile,
    footwear_contact: FootwearContactPreset,
    surface: str,
    modes: ModeSet,
    rng: np.random.Generator,
    sample_rate: int,
) -> FloatAudio:
    if grf.size == 0:
        return np.zeros(0, dtype=np.float32)
    floor_material = FLOOR_OBJECTS[surface].material
    time = np.arange(grf.size, dtype=np.float32) / np.float32(sample_rate)
    release_gate = np.clip((time - np.float32(0.055)) / np.float32(0.035), 0.0, 1.0)
    slip_velocity = release_gate * (np.float32(0.035) + np.float32(0.18) * grf)
    normal_load = grf * np.float32(620.0)
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
        viscous_coefficient_n_s_m=math.sqrt(
            footwear_contact.material.viscous_friction * floor_material.viscous_friction
        )
        * 0.08,
    )
    trace = lugre_friction(
        np.asarray(slip_velocity, dtype=np.float32),
        np.asarray(normal_load, dtype=np.float32),
        preset,
        sample_rate,
    )
    traveled = np.cumsum(np.abs(slip_velocity), dtype=np.float64) / sample_rate
    roughness_rms = math.sqrt(
        footwear_contact.material.roughness_rms_m**2 + floor_material.roughness_rms_m**2
    )
    correlation = math.sqrt(
        footwear_contact.material.roughness_correlation_m * floor_material.roughness_correlation_m
    )
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
    bank = ModalResonatorBank(modes, sample_rate)
    resonant = bank.process(excitation) * np.float32(90.0)
    direct = excitation * np.float32(
        0.025 * footwear.friction_gain * SOLID_SURFACE_PROFILES[surface].friction_roughness
    )
    result = np.asarray(direct + resonant, dtype=np.float32)
    if footwear.friction_brightness < 0.5:
        result = one_pole_lowpass(result, sample_rate, 2_200.0)
    else:
        result = one_pole_highpass(result, sample_rate, 480.0)
    return result


def _solid_footstep(
    *,
    grf: FloatAudio,
    surface: str,
    footwear: FootwearProfile,
    footwear_contact: FootwearContactPreset,
    rng: np.random.Generator,
    sample_rate: int,
) -> FloatAudio:
    """Render GRF-timed compliant contacts, floor modes, and loaded slip friction."""
    profile = SOLID_SURFACE_PROFILES[surface]
    size = grf.size
    if size == 0:
        return np.zeros(0, dtype=np.float32)
    physical_contact, floor_modes = _floor_contact_audio(
        grf=grf,
        surface=surface,
        footwear=footwear,
        footwear_contact=footwear_contact,
        rng=rng,
        sample_rate=sample_rate,
    )
    low_noise = bandpass_noise(rng, size, sample_rate, 42.0, 850.0)
    mid_noise = bandpass_noise(
        rng,
        size,
        sample_rate,
        max(100.0, profile.low_cut_hz),
        min(4_200.0, sample_rate * 0.44),
    )
    high_noise = bandpass_noise(
        rng,
        size,
        sample_rate,
        max(1_100.0, profile.low_cut_hz),
        min(profile.high_cut_hz, sample_rate * 0.46),
    )
    low_layer = low_noise * grf * np.float32(0.13 * footwear.low_weight_gain)
    mid_layer = mid_noise * grf * np.float32(0.17 * footwear.sole_gain)
    time = np.arange(size, dtype=np.float32) / np.float32(sample_rate)
    heel_env = asymmetric_pulse(
        size,
        sample_rate,
        attack_s=footwear.heel_attack_s,
        decay_s=max(0.002, min(0.024, footwear.heel_decay_s * 0.32)),
    )
    heel_max = float(np.max(heel_env)) if heel_env.size else 0.0
    if heel_max > 0.0:
        heel_env /= np.float32(heel_max)
    heel_noise = one_pole_highpass(high_noise, sample_rate, 420.0)
    heel_layer = (
        heel_noise * heel_env * np.float32(0.15 * footwear.heel_gain * profile.transient_gain)
    )
    contact_noise = np.asarray(
        physical_contact[:size] * np.float32(0.08) + low_layer + mid_layer + heel_layer,
        dtype=np.float32,
    )
    contact_noise = one_pole_lowpass(contact_noise, sample_rate, profile.high_cut_hz)
    contact_noise = one_pole_highpass(contact_noise, sample_rate, profile.low_cut_hz)
    output = physical_contact.copy()
    output[:size] += contact_noise
    friction = _mechanical_friction_audio(
        grf=grf,
        footwear=footwear,
        footwear_contact=footwear_contact,
        surface=surface,
        modes=floor_modes,
        rng=rng,
        sample_rate=sample_rate,
    )
    toe_start = int(rng.uniform(0.105, min(0.17, max(0.106, time[-1] - 0.015))) * sample_rate)
    toe_size = max(1, size - toe_start)
    scuff_noise = bandpass_noise(
        rng,
        toe_size,
        sample_rate,
        450.0,
        min(5_000.0, sample_rate * 0.45),
    )
    scuff_env = np.exp(
        -np.arange(toe_size, dtype=np.float32)
        / np.float32(max(0.018, footwear.toe_duration_s[1] * 0.55) * sample_rate)
    )
    friction[toe_start:] += (
        scuff_noise * scuff_env * np.float32(0.035 * footwear.toe_gain * profile.friction_roughness)
    )
    output[:size] += friction
    if surface == "wood":
        output = one_pole_lowpass(output, sample_rate, 900.0)
    elif surface == "carpet":
        output = one_pole_lowpass(output, sample_rate, 2_500.0)
    return np.asarray(output, dtype=np.float32)


def solid_footstep(
    *,
    sample_rate: int,
    surface: str,
    footwear: str,
    force: float,
    rng: np.random.Generator,
) -> FloatAudio:
    """Synthesize one GRF-driven solid-surface step."""
    profile = FOOTWEAR_PROFILES[footwear]
    grf = _synthetic_grf(
        sample_rate=sample_rate,
        footwear=footwear,
        force=force,
        rng=rng,
    )
    return _solid_footstep(
        grf=grf,
        surface=surface,
        footwear=profile,
        footwear_contact=FOOTWEAR_CONTACTS[footwear],
        rng=rng,
        sample_rate=sample_rate,
    )


@dataclass(frozen=True, slots=True)
class ParticleEvent:
    """One sampled aggregate micro-collision."""

    start_sample: int
    duration_samples: int
    energy: float
    center_hz: float
    resonant: bool

    work_budget_j: float


def _sample_particle_events(
    *,
    grf: FloatAudio,
    surface: AggregateSurfaceProfile,
    rng: np.random.Generator,
    sample_rate: int,
    footwear_contact: FootwearContactPreset | None = None,
    friction_gain: float = 0.5,
) -> list[ParticleEvent]:
    """Sample load-driven collisions and cap their energy by available slip work."""
    block_size = max(1, round(0.002 * sample_rate))
    low_hz, high_hz = surface.particle_band_hz
    events: list[ParticleEvent] = []
    material_mu = footwear_contact.material.friction_dynamic if footwear_contact else 0.78
    footwear_scale = 0.48 + 0.52 * friction_gain
    for block_start in range(0, grf.size, block_size):
        block_end = min(grf.size, block_start + block_size)
        block = grf[block_start:block_end]
        mean_load = max(0.0, float(np.mean(block, dtype=np.float64)))
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
            duration = max(2, round(float(rng.uniform(*surface.particle_duration_s)) * sample_rate))
            u = float(rng.random())
            proposed_energy = (
                surface.minimum_energy
                + (surface.maximum_energy - surface.minimum_energy) * u**surface.energy_power
            )
            energy = min(proposed_energy, remaining_work_j)
            if energy < surface.minimum_energy:
                break
            remaining_work_j -= energy
            center = math.exp(float(rng.uniform(math.log(low_hz), math.log(high_hz))))
            events.append(
                ParticleEvent(
                    start_sample=int(rng.integers(block_start, block_end)),
                    duration_samples=duration,
                    energy=energy,
                    center_hz=center,
                    resonant=bool(rng.random() < surface.resonance_probability),
                    work_budget_j=slip_work_j,
                )
            )
    return events


def _aggregate_footstep(
    *,
    grf: FloatAudio,
    surface: AggregateSurfaceProfile,
    footwear: FootwearProfile,
    footwear_contact: FootwearContactPreset,
    rng: np.random.Generator,
    sample_rate: int,
) -> FloatAudio:
    """Render load-bounded granular contacts over a physical hard-floor response."""
    if grf.size == 0:
        return np.zeros(0, dtype=np.float32)
    particle_tail = max(1, round(surface.particle_duration_s[1] * sample_rate))
    body, floor_modes = _floor_contact_audio(
        grf=grf,
        surface="stone",
        footwear=footwear,
        footwear_contact=footwear_contact,
        rng=rng,
        sample_rate=sample_rate,
    )
    output = np.zeros(max(body.size, grf.size + particle_tail), dtype=np.float32)
    output[: body.size] += body
    body_noise = bandpass_noise(rng, grf.size, sample_rate, 45.0, 1_400.0)
    output[: grf.size] += body_noise * grf * np.float32(0.13 * footwear.low_weight_gain)
    events = _sample_particle_events(
        grf=grf,
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
) -> FloatAudio:
    """Synthesize one seeded GRF-driven aggregate-surface step."""
    profile = FOOTWEAR_PROFILES[footwear]
    grf = _synthetic_grf(
        sample_rate=sample_rate,
        footwear=footwear,
        force=force,
        rng=rng,
    )
    return _aggregate_footstep(
        grf=grf,
        surface=AGGREGATE_SURFACE_PROFILES[surface],
        footwear=profile,
        footwear_contact=FOOTWEAR_CONTACTS[footwear],
        rng=rng,
        sample_rate=sample_rate,
    )

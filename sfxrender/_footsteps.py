"""GRF-driven footstep synthesis for solid and aggregate materials."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ._dsp import (
    asymmetric_pulse,
    bandpass_noise,
    impact_excitation,
    mix_at,
    modal_bank,
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


def _stable_modes(surface_name: str, sample_rate: int) -> tuple[tuple[float, float, float], ...]:
    """Sample stable base mode frequencies once per material, not per step."""
    profile = SOLID_SURFACE_PROFILES[surface_name]
    seeds = {"wood": 17_031, "stone": 28_019, "carpet": 39_011}
    material_rng = np.random.default_rng(seeds[surface_name])
    modes: list[tuple[float, float, float]] = []
    for index, band in enumerate(profile.mode_ranges):
        copies = 2 if surface_name == "wood" and index < 4 else 1
        for _ in range(copies):
            frequency = float(material_rng.uniform(band.low_hz, band.high_hz))
            if frequency >= sample_rate * 0.46:
                continue
            position_gain = float(material_rng.uniform(0.78, 1.16))
            gain = band.gain * position_gain * profile.resonance_gain
            modes.append((frequency, band.decay_s * profile.damping, gain))
    return tuple(modes)


def _solid_footstep(
    *,
    grf: FloatAudio,
    surface: str,
    footwear: FootwearProfile,
    rng: np.random.Generator,
    sample_rate: int,
) -> FloatAudio:
    """Excite a solid floor with aperiodic contact, then add weak modes/friction."""
    profile = SOLID_SURFACE_PROFILES[surface]
    size = grf.size
    if size == 0:
        return np.zeros(0, dtype=np.float32)

    contact = impact_excitation(
        sample_rate=sample_rate,
        duration=size,
        force_envelope=grf,
        hardness=footwear.heel_hardness,
        brightness=0.58 * profile.contact_brightness + 0.42 * footwear.friction_brightness,
        rng=rng,
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

    contact_mix = np.asarray(contact * np.float32(0.34) + low_layer + mid_layer + heel_layer)
    contact_mix = one_pole_lowpass(contact_mix, sample_rate, profile.high_cut_hz)
    contact_mix = one_pole_highpass(contact_mix, sample_rate, profile.low_cut_hz)
    modes = _stable_modes(surface, sample_rate)
    modal = modal_bank(
        contact_mix,
        sample_rate,
        rng,
        modes,
        frequency_jitter=0.009,
        decay_jitter=0.10,
    )
    modal *= np.float32(rng.uniform(0.86, 1.14))

    # Friction follows loaded movement and is deliberately quieter than contact.
    loaded_motion = np.maximum(grf, 0.0)
    release_gate = np.clip((time - np.float32(0.085)) / np.float32(0.035), 0.0, 1.0)
    friction_envelope = loaded_motion * (0.34 + 0.66 * release_gate)
    friction_noise = bandpass_noise(
        rng,
        size,
        sample_rate,
        180.0,
        min(4_800.0, sample_rate * 0.45),
    )
    if footwear.friction_brightness < 0.5:
        friction_noise = one_pole_lowpass(friction_noise, sample_rate, 2_200.0)
    else:
        friction_noise = one_pole_highpass(friction_noise, sample_rate, 480.0)
    friction = (
        friction_noise
        * friction_envelope
        * np.float32(0.055 * footwear.friction_gain * profile.friction_roughness)
    )

    # Very small, independently varied toe scuff adds a release cue without a pitched tone.
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

    return np.asarray(contact_mix + modal + friction, dtype=np.float32)


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


def _sample_particle_events(
    *,
    grf: FloatAudio,
    surface: AggregateSurfaceProfile,
    rng: np.random.Generator,
    sample_rate: int,
) -> list[ParticleEvent]:
    """Sample a blockwise Poisson process whose rate follows contact load."""
    block_size = max(1, round(0.002 * sample_rate))
    low_hz, high_hz = surface.particle_band_hz
    events: list[ParticleEvent] = []
    for block_start in range(0, grf.size, block_size):
        block_end = min(grf.size, block_start + block_size)
        block = grf[block_start:block_end]
        mean_load = max(0.0, float(np.mean(block, dtype=np.float64)))
        rate = surface.event_density * mean_load**surface.density_force_exponent
        count = int(rng.poisson(rate * (block_end - block_start) / sample_rate))
        for _ in range(count):
            duration = max(2, round(float(rng.uniform(*surface.particle_duration_s)) * sample_rate))
            u = float(rng.random())
            energy = (
                surface.minimum_energy
                + (surface.maximum_energy - surface.minimum_energy) * u**surface.energy_power
            )
            center = math.exp(float(rng.uniform(math.log(low_hz), math.log(high_hz))))
            events.append(
                ParticleEvent(
                    start_sample=int(rng.integers(block_start, block_end)),
                    duration_samples=duration,
                    energy=energy,
                    center_hz=center,
                    resonant=bool(rng.random() < surface.resonance_probability),
                )
            )
    return events


def _aggregate_footstep(
    *,
    grf: FloatAudio,
    surface: AggregateSurfaceProfile,
    footwear: FootwearProfile,
    rng: np.random.Generator,
    sample_rate: int,
) -> FloatAudio:
    """Render a granular surface as load-driven stochastic micro-impacts."""
    particle_tail = max(1, round(surface.particle_duration_s[1] * sample_rate))
    output = np.zeros(grf.size + particle_tail, dtype=np.float32)
    if grf.size == 0:
        return output
    force_peak = float(np.max(grf))
    body = impact_excitation(
        sample_rate=sample_rate,
        duration=grf.size,
        force_envelope=grf,
        hardness=0.28 + 0.42 * footwear.heel_hardness,
        brightness=0.58,
        rng=rng,
    )
    body_noise = bandpass_noise(rng, grf.size, sample_rate, 45.0, 1_400.0)
    body += body_noise * grf * np.float32(0.13 * footwear.low_weight_gain)
    output[: grf.size] += body * np.float32(0.30)

    events = _sample_particle_events(grf=grf, surface=surface, rng=rng, sample_rate=sample_rate)
    energy_scale = 0.62 + 0.60 * float(np.clip(force_peak, 0.0, 1.4))
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
        amplitude = math.sqrt(event.energy) * energy_scale * surface.texture_gain
        particle = noise * envelope * np.float32(amplitude)
        if event.resonant:
            phase = float(rng.uniform(-math.pi, math.pi))
            ring = np.sin(
                np.float32(2.0 * math.pi * event.center_hz) * t + np.float32(phase)
            ) * np.exp(-t / np.float32(max(0.001, decay * 0.7)))
            particle += ring * envelope * np.float32(amplitude * 0.075)
        mix_at(output, np.asarray(particle, dtype=np.float32), event.start_sample)
    slip_noise = bandpass_noise(
        rng,
        grf.size,
        sample_rate,
        260.0,
        min(4_200.0, sample_rate * 0.45),
    )
    movement = np.linspace(0.38, 1.0, grf.size, dtype=np.float32)
    output[: grf.size] += slip_noise * grf * movement * np.float32(0.022 * footwear.friction_gain)
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
        rng=rng,
        sample_rate=sample_rate,
    )

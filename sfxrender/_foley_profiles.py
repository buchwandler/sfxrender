"""Internal immutable parameter profiles for procedural footsteps and Foley."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ModeRange:
    """A resonant band and its nominal decay/gain."""

    low_hz: float
    high_hz: float
    gain: float
    decay_s: float


@dataclass(frozen=True, slots=True)
class FootwearProfile:
    """Exciter behavior for a foot/shoe, independent of floor identity."""

    heel_hardness: float
    heel_gain: float
    sole_gain: float
    toe_gain: float
    low_weight_gain: float
    heel_attack_s: float
    heel_decay_s: float
    sole_delay_s: tuple[float, float]
    sole_duration_s: tuple[float, float]
    toe_delay_s: tuple[float, float]
    toe_duration_s: tuple[float, float]
    friction_gain: float
    friction_brightness: float


@dataclass(frozen=True, slots=True)
class SolidSurfaceProfile:
    """Contact filtering and modal response of a solid walking surface."""

    contact_brightness: float
    transient_gain: float
    low_cut_hz: float
    high_cut_hz: float
    mode_ranges: tuple[ModeRange, ...]
    resonance_gain: float
    friction_roughness: float
    damping: float


@dataclass(frozen=True, slots=True)
class AggregateSurfaceProfile:
    """Stochastic micro-collision behavior for a granular surface."""

    event_density: float
    density_force_exponent: float
    energy_power: float
    minimum_energy: float
    maximum_energy: float
    particle_duration_s: tuple[float, float]
    particle_band_hz: tuple[float, float]
    resonance_probability: float
    texture_gain: float


FOOTWEAR_PROFILES: dict[str, FootwearProfile] = {
    "barefoot": FootwearProfile(
        heel_hardness=0.22,
        heel_gain=0.38,
        sole_gain=0.82,
        toe_gain=0.36,
        low_weight_gain=0.50,
        heel_attack_s=0.010,
        heel_decay_s=0.052,
        sole_delay_s=(0.034, 0.060),
        sole_duration_s=(0.064, 0.112),
        toe_delay_s=(0.112, 0.148),
        toe_duration_s=(0.046, 0.082),
        friction_gain=0.62,
        friction_brightness=0.30,
    ),
    "shoes": FootwearProfile(
        heel_hardness=0.56,
        heel_gain=0.68,
        sole_gain=0.78,
        toe_gain=0.32,
        low_weight_gain=0.56,
        heel_attack_s=0.0035,
        heel_decay_s=0.024,
        sole_delay_s=(0.032, 0.062),
        sole_duration_s=(0.050, 0.090),
        toe_delay_s=(0.108, 0.150),
        toe_duration_s=(0.040, 0.076),
        friction_gain=0.34,
        friction_brightness=0.58,
    ),
    "boots": FootwearProfile(
        heel_hardness=0.52,
        heel_gain=0.72,
        sole_gain=0.96,
        toe_gain=0.28,
        low_weight_gain=0.96,
        heel_attack_s=0.006,
        heel_decay_s=0.048,
        sole_delay_s=(0.040, 0.076),
        sole_duration_s=(0.078, 0.132),
        toe_delay_s=(0.128, 0.170),
        toe_duration_s=(0.052, 0.096),
        friction_gain=0.30,
        friction_brightness=0.42,
    ),
    "heels": FootwearProfile(
        heel_hardness=0.96,
        heel_gain=0.98,
        sole_gain=0.38,
        toe_gain=0.22,
        low_weight_gain=0.24,
        heel_attack_s=0.0018,
        heel_decay_s=0.013,
        sole_delay_s=(0.022, 0.042),
        sole_duration_s=(0.026, 0.052),
        toe_delay_s=(0.092, 0.130),
        toe_duration_s=(0.026, 0.058),
        friction_gain=0.18,
        friction_brightness=0.88,
    ),
}


SOLID_SURFACE_PROFILES: dict[str, SolidSurfaceProfile] = {
    "wood": SolidSurfaceProfile(
        contact_brightness=0.43,
        transient_gain=0.38,
        low_cut_hz=48.0,
        high_cut_hz=7_600.0,
        mode_ranges=(
            ModeRange(80.0, 180.0, 0.24, 0.095),
            ModeRange(170.0, 350.0, 0.23, 0.085),
            ModeRange(300.0, 650.0, 0.19, 0.067),
            ModeRange(500.0, 1_100.0, 0.15, 0.052),
            ModeRange(900.0, 1_800.0, 0.10, 0.038),
            ModeRange(1_400.0, 2_700.0, 0.07, 0.027),
        ),
        resonance_gain=0.85,
        friction_roughness=0.34,
        damping=0.88,
    ),
    "stone": SolidSurfaceProfile(
        contact_brightness=0.72,
        transient_gain=0.74,
        low_cut_hz=95.0,
        high_cut_hz=8_800.0,
        mode_ranges=(
            ModeRange(110.0, 240.0, 0.17, 0.052),
            ModeRange(230.0, 480.0, 0.16, 0.046),
            ModeRange(430.0, 850.0, 0.17, 0.039),
            ModeRange(760.0, 1_450.0, 0.14, 0.031),
            ModeRange(1_300.0, 2_300.0, 0.10, 0.023),
            ModeRange(2_000.0, 3_600.0, 0.07, 0.017),
        ),
        resonance_gain=0.10,
        friction_roughness=0.21,
        damping=0.62,
    ),
    "carpet": SolidSurfaceProfile(
        contact_brightness=0.16,
        transient_gain=0.10,
        low_cut_hz=38.0,
        high_cut_hz=3_600.0,
        mode_ranges=(
            ModeRange(70.0, 150.0, 0.07, 0.040),
            ModeRange(140.0, 280.0, 0.05, 0.030),
        ),
        resonance_gain=0.025,
        friction_roughness=0.12,
        damping=0.25,
    ),
}


AGGREGATE_SURFACE_PROFILES: dict[str, AggregateSurfaceProfile] = {
    "gravel": AggregateSurfaceProfile(
        event_density=340.0,
        density_force_exponent=0.65,
        energy_power=2.6,
        minimum_energy=0.012,
        maximum_energy=0.28,
        particle_duration_s=(0.003, 0.015),
        particle_band_hz=(700.0, 5_200.0),
        resonance_probability=0.14,
        texture_gain=0.58,
    ),
}


SOLID_SURFACES = frozenset(SOLID_SURFACE_PROFILES)

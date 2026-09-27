"""Perceptual generation ranges for wood and metal door models."""

from __future__ import annotations

from dataclasses import dataclass

from ._physics.models import ContactProfile, FrictionProfile, ModeBand


@dataclass(frozen=True, slots=True)
class DoorMaterialProfile:
    """Ranges and interaction traits used to generate one door identity."""

    panel_mode_bands: tuple[ModeBand, ...]
    frame_mode_bands: tuple[ModeBand, ...]
    latch_mode_bands: tuple[ModeBand, ...]
    hinge_friction: FrictionProfile
    handle_contact: ContactProfile
    latch_contact: ContactProfile
    frame_contact: ContactProfile
    panel_size_scale: tuple[float, float]
    damping_scale: tuple[float, float]


WOOD_DOOR_PROFILE = DoorMaterialProfile(
    panel_mode_bands=(
        ModeBand(70.0, 180.0, (2, 3), (0.20, 0.34), (0.10, 0.24)),
        ModeBand(150.0, 380.0, (2, 3), (0.19, 0.32), (0.09, 0.22)),
        ModeBand(300.0, 700.0, (2, 4), (0.14, 0.26), (0.06, 0.16)),
        ModeBand(600.0, 1_300.0, (1, 3), (0.08, 0.18), (0.035, 0.11)),
        ModeBand(1_200.0, 2_800.0, (1, 2), (0.025, 0.08), (0.018, 0.07)),
    ),
    frame_mode_bands=(
        ModeBand(100.0, 350.0, (2, 3), (0.17, 0.3), (0.055, 0.13)),
        ModeBand(300.0, 900.0, (2, 4), (0.12, 0.24), (0.035, 0.09)),
        ModeBand(800.0, 2_000.0, (1, 3), (0.04, 0.12), (0.018, 0.06)),
    ),
    latch_mode_bands=(
        ModeBand(500.0, 1_300.0, (1, 2), (0.12, 0.23), (0.018, 0.055)),
        ModeBand(1_200.0, 4_000.0, (2, 4), (0.08, 0.18), (0.008, 0.035)),
    ),
    hinge_friction=FrictionProfile(
        base_gain=0.42,
        roughness=0.42,
        noise_band_hz=(110.0, 2_200.0),
        stick_strength=0.66,
        slip_strength=0.46,
        f0_hz=(82.0, 205.0),
        harmonic_rolloff=(1.25, 2.0),
        chaos_amount=0.52,
    ),
    handle_contact=ContactProfile(0.42, 0.36, 0.34, 0.36),
    latch_contact=ContactProfile(0.58, 0.56, 0.42, 0.52),
    frame_contact=ContactProfile(0.48, 0.42, 0.34, 0.78),
    panel_size_scale=(0.88, 1.12),
    damping_scale=(0.78, 1.16),
)

METAL_DOOR_PROFILE = DoorMaterialProfile(
    panel_mode_bands=(
        ModeBand(150.0, 450.0, (2, 3), (0.18, 0.30), (0.13, 0.28)),
        ModeBand(350.0, 900.0, (2, 4), (0.15, 0.28), (0.12, 0.25)),
        ModeBand(700.0, 1_700.0, (2, 4), (0.13, 0.25), (0.10, 0.22)),
        ModeBand(1_500.0, 3_000.0, (2, 4), (0.10, 0.22), (0.08, 0.19)),
        ModeBand(2_500.0, 5_500.0, (2, 4), (0.08, 0.18), (0.07, 0.17)),
    ),
    frame_mode_bands=(
        ModeBand(180.0, 550.0, (2, 3), (0.17, 0.28), (0.12, 0.24)),
        ModeBand(450.0, 1_200.0, (2, 4), (0.15, 0.26), (0.10, 0.22)),
        ModeBand(1_000.0, 2_700.0, (2, 4), (0.11, 0.22), (0.075, 0.18)),
        ModeBand(2_200.0, 5_200.0, (1, 3), (0.08, 0.17), (0.065, 0.16)),
    ),
    latch_mode_bands=(
        ModeBand(650.0, 1_500.0, (1, 3), (0.16, 0.27), (0.07, 0.16)),
        ModeBand(1_400.0, 4_500.0, (2, 5), (0.13, 0.24), (0.06, 0.15)),
    ),
    hinge_friction=FrictionProfile(
        base_gain=0.36,
        roughness=0.62,
        noise_band_hz=(260.0, 4_800.0),
        stick_strength=0.55,
        slip_strength=0.72,
        f0_hz=(155.0, 370.0),
        harmonic_rolloff=(0.88, 1.5),
        chaos_amount=0.62,
    ),
    handle_contact=ContactProfile(0.72, 0.78, 0.48, 0.42),
    latch_contact=ContactProfile(0.82, 0.88, 0.62, 0.68),
    frame_contact=ContactProfile(0.78, 0.80, 0.54, 0.82),
    panel_size_scale=(0.92, 1.08),
    damping_scale=(0.98, 1.4),
)

DOOR_MATERIAL_PROFILES: dict[str, DoorMaterialProfile] = {
    "wood": WOOD_DOOR_PROFILE,
    "metal": METAL_DOOR_PROFILE,
}

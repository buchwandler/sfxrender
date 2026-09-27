"""Effective mechanical material presets for the private physics renderer."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MechanicalMaterial:
    """Physically motivated, calibration-friendly material parameters.

    Values describe an effective acoustic material rather than asserting exact
    textbook constants (particularly important for anisotropic wood products).
    """

    name: str
    density_kg_m3: float
    young_modulus_pa: float
    poisson_ratio: float
    loss_low: float
    loss_high: float
    loss_transition_hz: float
    restitution: float
    contact_hardness: float
    friction_static: float
    friction_dynamic: float
    stribeck_velocity_m_s: float
    viscous_friction: float
    roughness_rms_m: float
    roughness_correlation_m: float

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("name must not be empty")
        for name in (
            "density_kg_m3",
            "young_modulus_pa",
            "loss_transition_hz",
            "stribeck_velocity_m_s",
            "roughness_correlation_m",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        for name in (
            "loss_low",
            "loss_high",
            "friction_static",
            "friction_dynamic",
            "viscous_friction",
            "roughness_rms_m",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if not math.isfinite(self.poisson_ratio) or not -1.0 < self.poisson_ratio < 0.5:
            raise ValueError("poisson_ratio must be finite and between -1 and 0.5")
        if not 0.0 < self.restitution <= 1.0:
            raise ValueError("restitution must be greater than zero and at most one")
        if not 0.0 <= self.contact_hardness <= 1.0:
            raise ValueError("contact_hardness must be between zero and one")
        if self.friction_dynamic > self.friction_static:
            raise ValueError("friction_dynamic must not exceed friction_static")


OAK_EFFECTIVE = MechanicalMaterial(
    "oak_effective",
    680.0,
    1.05e10,
    0.31,
    0.045,
    0.075,
    900.0,
    0.42,
    0.58,
    0.62,
    0.48,
    0.035,
    0.025,
    1.8e-6,
    7.0e-4,
)
SOFTWOOD_EFFECTIVE = MechanicalMaterial(
    "softwood_effective",
    470.0,
    8.5e9,
    0.29,
    0.025,
    0.050,
    850.0,
    0.38,
    0.48,
    0.58,
    0.43,
    0.045,
    0.035,
    2.8e-6,
    1.1e-3,
)
STEEL_SHEET = MechanicalMaterial(
    "steel_sheet",
    7_850.0,
    2.0e11,
    0.30,
    0.010,
    0.025,
    1_500.0,
    0.72,
    0.94,
    0.55,
    0.42,
    0.025,
    0.012,
    0.7e-6,
    3.0e-4,
)
PAINTED_STEEL = MechanicalMaterial(
    "painted_steel",
    7_700.0,
    1.85e11,
    0.30,
    0.003,
    0.009,
    1_200.0,
    0.58,
    0.84,
    0.62,
    0.48,
    0.035,
    0.020,
    1.2e-6,
    5.0e-4,
)
DRYWALL_PANEL = MechanicalMaterial(
    "drywall_panel",
    760.0,
    3.2e9,
    0.24,
    0.14,
    0.22,
    700.0,
    0.28,
    0.36,
    0.72,
    0.58,
    0.055,
    0.045,
    3.5e-6,
    1.4e-3,
)
STONE_TILE = MechanicalMaterial(
    "stone_tile",
    2_350.0,
    5.2e10,
    0.22,
    0.006,
    0.015,
    1_300.0,
    0.34,
    0.88,
    0.78,
    0.64,
    0.030,
    0.018,
    0.5e-6,
    2.5e-4,
)
RUBBER_SOLE = MechanicalMaterial(
    "rubber_sole",
    1_100.0,
    1.2e7,
    0.48,
    0.12,
    0.22,
    400.0,
    0.22,
    0.18,
    0.92,
    0.72,
    0.080,
    0.12,
    8.0e-6,
    2.0e-3,
)
LEATHER_SOLE = MechanicalMaterial(
    "leather_sole",
    950.0,
    5.0e8,
    0.35,
    0.055,
    0.11,
    550.0,
    0.32,
    0.48,
    0.74,
    0.56,
    0.050,
    0.055,
    4.0e-6,
    1.0e-3,
)

MATERIALS: dict[str, MechanicalMaterial] = {
    material.name: material
    for material in (
        OAK_EFFECTIVE,
        SOFTWOOD_EFFECTIVE,
        STEEL_SHEET,
        PAINTED_STEEL,
        DRYWALL_PANEL,
        STONE_TILE,
        RUBBER_SOLE,
        LEATHER_SOLE,
    )
}

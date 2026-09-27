"""Private acoustic-object presets assembled from effective materials/geometry."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .geometry import BoundaryCondition, RectangularPlate
from .materials import (
    DRYWALL_PANEL,
    LEATHER_SOLE,
    OAK_EFFECTIVE,
    PAINTED_STEEL,
    RUBBER_SOLE,
    SOFTWOOD_EFFECTIVE,
    STEEL_SHEET,
    STONE_TILE,
    MechanicalMaterial,
)
from .modes import ModeSet


@dataclass(frozen=True, slots=True)
class AcousticObjectPreset:
    name: str
    material: MechanicalMaterial
    geometry: RectangularPlate
    boundary: BoundaryCondition
    modes: ModeSet | None = None

    mode_decay_scale: float = 1.0
    modal_gain: float = 1.0
    modal_mass_scale: float = 1.0

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("acoustic object name must not be empty")
        if self.geometry.boundary is not self.boundary:
            raise ValueError("geometry and acoustic object boundary conditions must match")
        if not math.isfinite(self.mode_decay_scale) or self.mode_decay_scale <= 0.0:
            raise ValueError("mode_decay_scale must be finite and positive")
        if not math.isfinite(self.modal_gain) or self.modal_gain < 0.0:
            raise ValueError("modal_gain must be finite and non-negative")
        if not math.isfinite(self.modal_mass_scale) or self.modal_mass_scale <= 0.0:
            raise ValueError("modal_mass_scale must be finite and positive")


@dataclass(frozen=True, slots=True)
class ImpactorPreset:
    name: str
    effective_mass_kg: float
    stiffness: float
    restitution: float
    microscopic_gain: float
    default_x: float = 0.62
    default_y: float = 0.54


@dataclass(frozen=True, slots=True)
class FootwearContactPreset:
    """Effective impactor mechanics for one footwear class."""

    name: str
    material: MechanicalMaterial
    effective_mass_kg: float
    normal_stiffness_n_m: float
    contact_area_m2: float

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("footwear preset name must not be empty")
        for name in ("effective_mass_kg", "normal_stiffness_n_m", "contact_area_m2"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")


FOOTWEAR_CONTACTS: dict[str, FootwearContactPreset] = {
    "barefoot": FootwearContactPreset("barefoot", RUBBER_SOLE, 0.16, 190_000.0, 0.013),
    "shoes": FootwearContactPreset("shoes", LEATHER_SOLE, 0.30, 1_200_000.0, 0.006),
    "boots": FootwearContactPreset("boots", RUBBER_SOLE, 0.48, 650_000.0, 0.015),
    "heels": FootwearContactPreset("heels", STEEL_SHEET, 0.11, 4_800_000.0, 0.00045),
}


FLOOR_OBJECTS: dict[str, AcousticObjectPreset] = {
    "wood": AcousticObjectPreset(
        "wood_floor",
        SOFTWOOD_EFFECTIVE,
        RectangularPlate(2.4, 3.6, 0.028, BoundaryCondition.FLOOR_SUPPORTED),
        BoundaryCondition.FLOOR_SUPPORTED,
        mode_decay_scale=0.88,
        modal_gain=2.5,
        modal_mass_scale=0.12,
    ),
    "stone": AcousticObjectPreset(
        "stone_tile_floor",
        STONE_TILE,
        RectangularPlate(0.60, 0.60, 0.025, BoundaryCondition.FLOOR_SUPPORTED),
        BoundaryCondition.FLOOR_SUPPORTED,
        mode_decay_scale=0.72,
        modal_gain=0.72,
    ),
    "carpet": AcousticObjectPreset(
        "carpeted_subfloor",
        SOFTWOOD_EFFECTIVE,
        RectangularPlate(1.2, 1.8, 0.022, BoundaryCondition.FLOOR_SUPPORTED),
        BoundaryCondition.FLOOR_SUPPORTED,
        mode_decay_scale=0.16,
        modal_gain=0.16,
    ),
}

KNOCK_IMPACTORS: dict[str, ImpactorPreset] = {
    "fingertip": ImpactorPreset("fingertip", 0.012, 160_000.0, 0.24, 0.45, 0.55, 0.50),
    "knuckle": ImpactorPreset("knuckle", 0.035, 1_400_000.0, 0.38, 0.75),
    "wooden_object": ImpactorPreset("wooden_object", 0.080, 5_000_000.0, 0.48, 0.60),
    "metal_object": ImpactorPreset("metal_object", 0.024, 8_000_000.0, 0.64, 0.95),
}

KNOCK_OBJECTS: dict[str, str] = {
    "wood": "small_wood_board",
    "oak": "oak_door",
    "wall": "drywall_panel",
    "metal": "steel_sheet",
}

OBJECT_PRESETS: dict[str, AcousticObjectPreset] = {
    "small_wood_board": AcousticObjectPreset(
        "small_wood_board",
        SOFTWOOD_EFFECTIVE,
        RectangularPlate(0.28, 0.20, 0.018),
        BoundaryCondition.SIMPLY_SUPPORTED,
    ),
    "oak_door": AcousticObjectPreset(
        "oak_door",
        OAK_EFFECTIVE,
        RectangularPlate(0.86, 2.03, 0.040, BoundaryCondition.DOOR_HINGED),
        BoundaryCondition.DOOR_HINGED,
    ),
    "drywall_panel": AcousticObjectPreset(
        "drywall_panel",
        DRYWALL_PANEL,
        RectangularPlate(0.40, 0.80, 0.013, BoundaryCondition.WALL_PANEL),
        BoundaryCondition.WALL_PANEL,
    ),
    "steel_sheet": AcousticObjectPreset(
        "steel_sheet",
        STEEL_SHEET,
        RectangularPlate(0.50, 0.70, 0.0012, BoundaryCondition.CLAMPED),
        BoundaryCondition.CLAMPED,
    ),
    "painted_steel_door": AcousticObjectPreset(
        "painted_steel_door",
        PAINTED_STEEL,
        RectangularPlate(0.90, 2.05, 0.032, BoundaryCondition.DOOR_HINGED),
        BoundaryCondition.DOOR_HINGED,
    ),
    "stone_tile": AcousticObjectPreset(
        "stone_tile",
        STONE_TILE,
        RectangularPlate(0.60, 0.60, 0.025, BoundaryCondition.FLOOR_SUPPORTED),
        BoundaryCondition.FLOOR_SUPPORTED,
    ),
}

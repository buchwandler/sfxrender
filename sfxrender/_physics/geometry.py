"""Analytic effective geometry and thin-plate modal approximations."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache

from .materials import MechanicalMaterial
from .modes import Mode, ModeSet
from .radiation import radiation_efficiency


class BoundaryCondition(str, Enum):
    SIMPLY_SUPPORTED = "simply_supported"
    CLAMPED = "clamped"
    FREE = "free"
    DOOR_HINGED = "door_hinged"
    FLOOR_SUPPORTED = "floor_supported"
    WALL_PANEL = "wall_panel"


@dataclass(frozen=True, slots=True)
class RectangularPlate:
    width_m: float
    height_m: float
    thickness_m: float
    boundary: BoundaryCondition | str = BoundaryCondition.SIMPLY_SUPPORTED

    def __post_init__(self) -> None:
        for name in ("width_m", "height_m", "thickness_m"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        try:
            condition = BoundaryCondition(self.boundary)
        except ValueError as error:
            raise ValueError(f"unsupported boundary condition: {self.boundary}") from error
        object.__setattr__(self, "boundary", condition)


@dataclass(frozen=True, slots=True)
class LumpedBody:
    mass_kg: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.mass_kg) or self.mass_kg <= 0.0:
            raise ValueError("mass_kg must be finite and positive")


@dataclass(frozen=True, slots=True)
class BoundaryPreset:
    frequency_scale: float
    damping_scale: float
    low_mode_gain: float = 1.0


_BOUNDARY_PRESETS: dict[BoundaryCondition, BoundaryPreset] = {
    BoundaryCondition.SIMPLY_SUPPORTED: BoundaryPreset(1.0, 1.0, 1.0),
    BoundaryCondition.CLAMPED: BoundaryPreset(1.32, 1.15, 0.90),
    BoundaryCondition.FREE: BoundaryPreset(0.72, 0.72, 1.15),
    BoundaryCondition.DOOR_HINGED: BoundaryPreset(1.08, 1.18, 0.82),
    BoundaryCondition.FLOOR_SUPPORTED: BoundaryPreset(0.91, 1.45, 0.74),
    BoundaryCondition.WALL_PANEL: BoundaryPreset(1.16, 1.35, 0.78),
}


def plate_mode_shape(m: int, n: int, x: float, y: float) -> float:
    """Evaluate a simply-supported rectangular plate's normalized mode shape."""
    if m < 1 or n < 1:
        raise ValueError("plate mode indices must be positive")
    if not math.isfinite(x) or not 0.0 <= x <= 1.0:
        raise ValueError("x must be finite and between zero and one")
    if not math.isfinite(y) or not 0.0 <= y <= 1.0:
        raise ValueError("y must be finite and between zero and one")
    return math.sin(m * math.pi * x) * math.sin(n * math.pi * y)


def modal_decay(material: MechanicalMaterial, frequency_hz: float) -> float:
    """Interpolate material loss by log-frequency and return amplitude decay time."""
    if not math.isfinite(frequency_hz) or frequency_hz <= 0.0:
        raise ValueError("frequency_hz must be finite and positive")
    position = math.log(max(frequency_hz, 1e-9) / material.loss_transition_hz)
    blend = 0.5 + 0.5 * math.tanh(position * 1.5)
    loss = material.loss_low + (material.loss_high - material.loss_low) * blend
    loss = max(loss, 1e-5)
    q = 1.0 / loss
    return q / (math.pi * frequency_hz)


@lru_cache(maxsize=128)
def _cached_mode_indices(
    width_m: float,
    height_m: float,
    thickness_m: float,
    young_modulus_pa: float,
    poisson_ratio: float,
    density_kg_m3: float,
    frequency_scale: float,
    max_frequency_hz: float,
    max_modes: int,
) -> tuple[tuple[int, int, float], ...]:
    rigidity = young_modulus_pa * thickness_m**3 / (12.0 * (1.0 - poisson_ratio**2))
    wave_scale = math.sqrt(rigidity / (density_kg_m3 * thickness_m))
    max_m = max(
        1, int(width_m * math.sqrt(max_frequency_hz * 2.0 * math.pi / wave_scale) / math.pi) + 2
    )
    max_n = max(
        1, int(height_m * math.sqrt(max_frequency_hz * 2.0 * math.pi / wave_scale) / math.pi) + 2
    )
    candidates: list[tuple[int, int, float]] = []
    for m in range(1, max_m + 1):
        for n in range(1, max_n + 1):
            omega = wave_scale * ((m * math.pi / width_m) ** 2 + (n * math.pi / height_m) ** 2)
            frequency = omega * frequency_scale / (2.0 * math.pi)
            if frequency < max_frequency_hz:
                candidates.append((m, n, frequency))
    candidates.sort(key=lambda item: item[2])
    return tuple(candidates)


def rectangular_plate_modes(
    plate: RectangularPlate,
    material: MechanicalMaterial,
    *,
    sample_rate: int,
    max_frequency_hz: float | None = None,
    max_modes: int = 32,
    contact_x: float = 0.5,
    contact_y: float = 0.5,
) -> ModeSet:
    """Generate approximate bending modes for an effective thin rectangular plate."""
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if max_modes <= 0:
        raise ValueError("max_modes must be positive")
    if not 0.0 <= contact_x <= 1.0 or not math.isfinite(contact_x):
        raise ValueError("contact_x must be finite and between zero and one")
    if not 0.0 <= contact_y <= 1.0 or not math.isfinite(contact_y):
        raise ValueError("contact_y must be finite and between zero and one")
    upper = (
        sample_rate * 0.45
        if max_frequency_hz is None
        else min(max_frequency_hz, sample_rate * 0.45)
    )
    if not math.isfinite(upper) or upper <= 0.0:
        raise ValueError("max_frequency_hz must be finite and positive")
    boundary = _BOUNDARY_PRESETS[BoundaryCondition(plate.boundary)]
    candidates = _cached_mode_indices(
        plate.width_m,
        plate.height_m,
        plate.thickness_m,
        material.young_modulus_pa,
        material.poisson_ratio,
        material.density_kg_m3,
        boundary.frequency_scale,
        upper,
        max_modes,
    )
    if len(candidates) > max_modes:
        low_count = min(max_modes, max(1, max_modes // 4))
        remaining = max_modes - low_count
        indices = list(range(low_count))
        if remaining == 1:
            indices.append(len(candidates) - 1)
        elif remaining > 1:
            span = len(candidates) - 1 - low_count
            step = span / (remaining - 1)
            indices.extend(round(low_count + index * step) for index in range(remaining))
        candidates = tuple(candidates[index] for index in indices)
    modal_mass = material.density_kg_m3 * plate.thickness_m * plate.width_m * plate.height_m / 4.0
    characteristic_size = math.sqrt(plate.width_m * plate.height_m)
    modes = tuple(
        Mode(
            frequency_hz=frequency,
            decay_s=modal_decay(material, frequency) / boundary.damping_scale,
            input_gain=plate_mode_shape(m, n, contact_x, contact_y),
            radiation_gain=(
                radiation_efficiency(frequency, characteristic_size)
                * boundary.low_mode_gain
                * min(1.0, (frequency / 500.0) ** 3.5)
            ),
            modal_mass_kg=modal_mass,
        )
        for m, n, frequency in candidates
    )
    return ModeSet(modes)


plate_modes = rectangular_plate_modes

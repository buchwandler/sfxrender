"""Oversampled compliant normal-impact contact traces."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..types import FloatAudio


@dataclass(frozen=True, slots=True)
class ImpactContact:
    effective_mass_kg: float
    stiffness: float
    exponent: float = 1.5
    restitution: float = 0.45

    def __post_init__(self) -> None:
        if not math.isfinite(self.effective_mass_kg) or self.effective_mass_kg <= 0.0:
            raise ValueError("effective_mass_kg must be finite and positive")
        if not math.isfinite(self.stiffness) or self.stiffness <= 0.0:
            raise ValueError("stiffness must be finite and positive")
        if not math.isfinite(self.exponent) or not 1.0 <= self.exponent <= 2.0:
            raise ValueError("exponent must be finite and between one and two")
        if not math.isfinite(self.restitution) or not 0.0 < self.restitution <= 1.0:
            raise ValueError("restitution must be greater than zero and at most one")


@dataclass(frozen=True, slots=True)
class ImpactEvent:
    velocity_m_s: float
    contact: ImpactContact

    def __post_init__(self) -> None:
        if not math.isfinite(self.velocity_m_s) or self.velocity_m_s < 0.0:
            raise ValueError("velocity_m_s must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class ContactTrace:
    force_n: FloatAudio
    indentation_m: FloatAudio
    velocity_m_s: FloatAudio
    rebound_velocity_m_s: float


def _downsample_mean(values: list[float], factor: int) -> FloatAudio:
    if not values:
        return np.zeros(0, dtype=np.float32)
    array = np.asarray(values, dtype=np.float64)
    padded_size = math.ceil(array.size / factor) * factor
    if padded_size != array.size:
        array = np.pad(array, (0, padded_size - array.size))
    return np.asarray(array.reshape(-1, factor).mean(axis=1), dtype=np.float32)


def impact_force(
    *,
    contact: ImpactContact,
    velocity_m_s: float,
    sample_rate: int,
    oversample: int = 4,
    max_contact_s: float = 0.12,
) -> ContactTrace:
    """Integrate a Hunt-Crossley contact and return its audio-rate force trace.

    The effective compression coordinate obeys ``m * delta_ddot = -F`` and
    ``F = k * delta**p * max(0, 1 + d * delta_dot)``. A restitution-derived
    damping estimate and conservative substeps provide useful first-order
    calibration without introducing an external physics dependency.
    """
    ImpactEvent(velocity_m_s, contact)
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if not isinstance(oversample, int) or oversample < 1:
        raise ValueError("oversample must be a positive integer")
    if not math.isfinite(max_contact_s) or max_contact_s <= 0.0:
        raise ValueError("max_contact_s must be finite and positive")
    if velocity_m_s == 0.0:
        empty = np.zeros(1, dtype=np.float32)
        return ContactTrace(empty, empty.copy(), empty.copy(), 0.0)

    energy = 0.5 * contact.effective_mass_kg * velocity_m_s**2
    estimated_peak_indent = (energy * (contact.exponent + 1.0) / contact.stiffness) ** (
        1.0 / (contact.exponent + 1.0)
    )
    linearized_stiffness = (
        contact.exponent
        * contact.stiffness
        * max(estimated_peak_indent, 1e-9) ** (contact.exponent - 1.0)
    )
    natural_frequency = math.sqrt(linearized_stiffness / contact.effective_mass_kg)
    factor = max(
        oversample,
        math.ceil(96_000 / sample_rate),
        math.ceil(20.0 * natural_frequency / sample_rate),
    )
    factor = min(factor, 256)
    physics_rate = sample_rate * factor
    dt = 1.0 / physics_rate
    damping = 1.125 * (1.0 - contact.restitution**2) / max(velocity_m_s, 1e-6)
    max_indent = min(max(estimated_peak_indent * 4.0, 1e-5), 0.05)
    max_force = min(max(4.0 * energy / max_indent, 1.0), 1e8)
    max_steps = max(2, math.ceil(max_contact_s * physics_rate))

    indentation = 0.0
    compression_velocity = velocity_m_s
    forces: list[float] = []
    indentations: list[float] = []
    velocities: list[float] = []
    rebound_velocity = 0.0
    contact_ended = False
    for _ in range(max_steps):
        positive_indent = min(max(indentation, 0.0), max_indent)
        damping_factor = max(0.0, 1.0 + damping * compression_velocity)
        force = min(
            contact.stiffness * positive_indent**contact.exponent * damping_factor,
            max_force,
        )
        if not math.isfinite(force):
            raise FloatingPointError("impact contact produced a non-finite force")
        forces.append(max(force, 0.0))
        indentations.append(positive_indent)
        velocities.append(compression_velocity)
        next_velocity = compression_velocity - force * dt / contact.effective_mass_kg
        next_indentation = indentation + next_velocity * dt
        if next_indentation <= 0.0 and next_velocity < 0.0:
            rebound_velocity = -next_velocity
            forces.append(0.0)
            indentations.append(0.0)
            velocities.append(next_velocity)
            contact_ended = True
            break
        indentation = min(next_indentation, max_indent)
        compression_velocity = next_velocity

    if not contact_ended:
        # A bounded trace is preferable to an unbounded integration if a hostile
        # parameter set fails to separate within the requested contact window.
        forces.append(0.0)
        indentations.append(0.0)
        velocities.append(compression_velocity)

    force_audio = _downsample_mean(forces, factor)
    indentation_audio = _downsample_mean(indentations, factor)
    velocity_audio = _downsample_mean(velocities, factor)
    if force_audio.size:
        force_audio[0] = 0.0
        force_audio[-1] = 0.0
    return ContactTrace(
        force_n=force_audio,
        indentation_m=indentation_audio,
        velocity_m_s=velocity_audio,
        rebound_velocity_m_s=float(rebound_velocity),
    )

"""Reusable struck-resonator and mechanical strike-train rendering."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ..types import FloatAudio
from .contact import ImpactContact, impact_force
from .modes import ModeSet
from .physical_impacts import render_force_response


@dataclass(frozen=True, slots=True)
class StrikeEvent:
    """A scheduled strike assigned to one resonator."""

    time_s: float
    velocity_m_s: float
    resonator_index: int = 0

    def __post_init__(self) -> None:
        if not math.isfinite(self.time_s) or self.time_s < 0.0:
            raise ValueError("time_s must be finite and non-negative")
        if not math.isfinite(self.velocity_m_s) or self.velocity_m_s < 0.0:
            raise ValueError("velocity_m_s must be finite and non-negative")
        if isinstance(self.resonator_index, bool) or not isinstance(self.resonator_index, int):
            raise TypeError("resonator_index must be an integer")
        if self.resonator_index < 0:
            raise ValueError("resonator_index must be non-negative")


@dataclass(frozen=True, slots=True)
class StruckResonatorPreset:
    """Contact and modal identity for one mechanically struck resonator."""

    modes: ModeSet
    contact: ImpactContact
    microscopic_gain: float = 0.4
    output_gain: float = 24.0

    def __post_init__(self) -> None:
        if not isinstance(self.modes, ModeSet):
            raise TypeError("modes must be a ModeSet")
        if not isinstance(self.contact, ImpactContact):
            raise TypeError("contact must be an ImpactContact")
        if not math.isfinite(self.microscopic_gain) or self.microscopic_gain < 0.0:
            raise ValueError("microscopic_gain must be finite and non-negative")
        if not math.isfinite(self.output_gain) or self.output_gain < 0.0:
            raise ValueError("output_gain must be finite and non-negative")


def build_contact_force_bus(
    events: Sequence[StrikeEvent],
    *,
    resonator_index: int,
    contact: ImpactContact,
    duration_s: float,
    sample_rate: int,
) -> FloatAudio:
    """Accumulate scheduled contact traces for one resonator on one timeline."""
    if isinstance(resonator_index, bool) or not isinstance(resonator_index, int):
        raise TypeError("resonator_index must be an integer")
    if resonator_index < 0:
        raise ValueError("resonator_index must be non-negative")
    if not isinstance(contact, ImpactContact):
        raise TypeError("contact must be an ImpactContact")
    if not math.isfinite(duration_s) or duration_s <= 0.0:
        raise ValueError("duration_s must be finite and positive")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    bus = np.zeros(max(1, round(duration_s * sample_rate)), dtype=np.float32)
    duration_samples = bus.size
    for event in events:
        if not isinstance(event, StrikeEvent):
            raise TypeError("events must contain StrikeEvent values")
        start = round(event.time_s * sample_rate)
        if event.resonator_index != resonator_index or start >= duration_samples:
            continue
        trace = impact_force(
            contact=contact,
            velocity_m_s=event.velocity_m_s,
            sample_rate=sample_rate,
        )
        end = min(duration_samples, start + trace.force_n.size)
        if end > start:
            bus[start:end] += trace.force_n[: end - start]
    return bus


def render_strike_train(
    *,
    events: Sequence[StrikeEvent],
    resonators: Sequence[StruckResonatorPreset],
    duration_s: float,
    sample_rate: int,
    rngs: Sequence[np.random.Generator],
) -> FloatAudio:
    """Render all strikes by accumulating one force bus per resonator."""
    if not math.isfinite(duration_s) or duration_s <= 0.0:
        raise ValueError("duration_s must be finite and positive")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if len(rngs) != len(resonators):
        raise ValueError("rngs must contain one generator per resonator")
    if any(not isinstance(rng, np.random.Generator) for rng in rngs):
        raise TypeError("rngs must contain NumPy Generator values")
    if any(not isinstance(preset, StruckResonatorPreset) for preset in resonators):
        raise TypeError("resonators must contain StruckResonatorPreset values")
    if any(not isinstance(event, StrikeEvent) for event in events):
        raise TypeError("events must contain StrikeEvent values")
    if any(event.resonator_index >= len(resonators) for event in events):
        raise ValueError("strike event selects a missing resonator")

    rendered: list[FloatAudio] = []
    for index, (preset, rng) in enumerate(zip(resonators, rngs, strict=True)):
        force_bus = build_contact_force_bus(
            events,
            resonator_index=index,
            contact=preset.contact,
            duration_s=duration_s,
            sample_rate=sample_rate,
        )
        rendered.append(
            render_force_response(
                force_bus,
                preset.modes,
                sample_rate=sample_rate,
                rng=rng,
                microscopic_gain=preset.microscopic_gain,
                output_gain=preset.output_gain,
            )
        )
    output_size = round(duration_s * sample_rate)
    output = np.zeros(max(output_size, *(samples.size for samples in rendered)), dtype=np.float32)
    for samples in rendered:
        output[: samples.size] += samples
    return output

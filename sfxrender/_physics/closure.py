"""Terminal contact responses for objects reaching a mechanical stop."""

from __future__ import annotations

import math

from ..types import FloatAudio
from .contact import ImpactContact, impact_force
from .modes import ModeSet
from .physical_impacts import render_force_response
from .rng import component_rng

_CLOSURE_NAMESPACE = 0x434C4F53


def render_terminal_closure(
    *,
    contact: ImpactContact,
    velocity_m_s: float,
    modes: ModeSet,
    sample_rate: int,
    seed: int,
    event_index: int = 0,
    output_gain: float = 20.0,
    microscopic_gain: float = 0.12,
    tail_s: float = 0.18,
) -> FloatAudio:
    """Render one compliant terminal impact through its target object's modes."""
    if not math.isfinite(velocity_m_s) or velocity_m_s < 0.0:
        raise ValueError("velocity_m_s must be finite and non-negative")
    if not math.isfinite(tail_s) or tail_s < 0.0:
        raise ValueError("tail_s must be finite and non-negative")
    trace = impact_force(
        contact=contact,
        velocity_m_s=velocity_m_s,
        sample_rate=sample_rate,
    )
    return render_force_response(
        trace.force_n,
        modes,
        sample_rate=sample_rate,
        rng=component_rng(seed, _CLOSURE_NAMESPACE, 1, event_index),
        microscopic_gain=microscopic_gain,
        output_gain=output_gain,
        tail_s=tail_s,
    )

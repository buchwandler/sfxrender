"""Synthesis helpers for physical contact traces exciting acoustic objects."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .._dsp import bandpass_noise
from ..types import FloatAudio
from .contact import ContactTrace
from .modes import ModeSet
from .resonator import modal_response


def render_force_response(
    force_n: FloatAudio,
    modes: ModeSet,
    *,
    sample_rate: int,
    rng: np.random.Generator,
    microscopic_gain: float = 0.5,
    output_gain: float = 24.0,
    tail_s: float | None = None,
) -> FloatAudio:
    """Render a finite mono force signal through structural modes."""
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if not math.isfinite(microscopic_gain) or microscopic_gain < 0.0:
        raise ValueError("microscopic_gain must be finite and non-negative")
    if not math.isfinite(output_gain) or output_gain < 0.0:
        raise ValueError("output_gain must be finite and non-negative")
    force = np.asarray(force_n, dtype=np.float32)
    if force.ndim != 1 or not np.all(np.isfinite(force)) or np.any(force < 0.0):
        raise ValueError("force must be a finite non-negative mono signal")
    if force.size == 0 or not modes.modes or not np.any(force):
        return np.zeros(force.size, dtype=np.float32)
    if tail_s is None:
        tail_s = min(1.25, max(mode.decay_s for mode in modes.modes) * 4.5)
    elif not math.isfinite(tail_s) or tail_s < 0.0:
        raise ValueError("tail_s must be finite and non-negative")

    excitation = force
    texture: FloatAudio | None = None
    if microscopic_gain > 0.0:
        high = min(sample_rate * 0.46, 9_000.0)
        low = min(max(120.0, high * 0.18), max(1.0, high - 1.0))
        texture = bandpass_noise(rng, force.size, sample_rate, low, high)
        micro_force = texture * force * np.float32(0.08 * microscopic_gain)
        excitation = np.asarray(force + micro_force, dtype=np.float32)

    response = modal_response(excitation, modes, sample_rate, tail_s=tail_s)
    response *= np.float32(output_gain)
    if texture is not None:
        micro_audio = texture * force * np.float32(0.008 * microscopic_gain * output_gain)
        response[: force.size] += micro_audio
    return np.asarray(response, dtype=np.float32)


def render_physical_impact(
    trace: ContactTrace,
    modes: ModeSet,
    *,
    sample_rate: int,
    rng: np.random.Generator,
    microscopic_gain: float = 0.5,
    output_gain: float = 24.0,
) -> FloatAudio:
    """Compatibility wrapper rendering a contact trace through object modes."""
    return render_force_response(
        trace.force_n,
        modes,
        sample_rate=sample_rate,
        rng=rng,
        microscopic_gain=microscopic_gain,
        output_gain=output_gain,
    )


@dataclass(frozen=True, slots=True)
class ImpactResponse:
    """One structural response driven by a shared impact force trace."""

    modes: ModeSet
    gain: float = 1.0
    microscopic_gain: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.modes, ModeSet):
            raise TypeError("modes must be a ModeSet")
        if not math.isfinite(self.gain) or self.gain < 0.0:
            raise ValueError("gain must be finite and non-negative")
        if not math.isfinite(self.microscopic_gain) or self.microscopic_gain < 0.0:
            raise ValueError("microscopic_gain must be finite and non-negative")


def render_coupled_impact(
    trace: ContactTrace,
    responses: Sequence[ImpactResponse],
    *,
    sample_rate: int,
    rngs: Sequence[np.random.Generator],
) -> FloatAudio:
    """Render one contact force trace through multiple structural responses."""
    if len(rngs) != len(responses):
        raise ValueError("rngs must contain one generator per response")
    if not responses:
        return np.zeros(np.asarray(trace.force_n).size, dtype=np.float32)

    rendered = [
        render_force_response(
            trace.force_n,
            response.modes,
            sample_rate=sample_rate,
            rng=rng,
            microscopic_gain=response.microscopic_gain,
            output_gain=24.0 * response.gain,
        )
        for response, rng in zip(responses, rngs, strict=True)
    ]
    output = np.zeros(max(samples.size for samples in rendered), dtype=np.float32)
    for samples in rendered:
        output[: samples.size] += samples
    return output

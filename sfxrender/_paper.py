"""Semantic page-turn and paper-handling actions."""

from __future__ import annotations

import numpy as np

from ._dsp import envelope_from_points, mix_at
from ._physics.rng import component_rng
from ._physics.thin_material import (
    generate_thin_material_preset,
    render_thin_material_source,
)
from .types import FloatAudio

_PAPER_NAMESPACE = 0x50415052
_TURN_DURATION = {"slow": 0.68, "normal": 0.48, "fast": 0.32}


def render_page_turn(*, sample_rate: int, pages: int, speed: str, seed: int) -> FloatAudio:
    preset = generate_thin_material_preset(seed=seed, sample_rate=sample_rate)
    duration = _TURN_DURATION[speed]
    gesture_size = max(1, round(duration * sample_rate))
    activity = envelope_from_points(
        [
            (0.0, 0.0),
            (0.08 * duration, 0.24),
            (0.24 * duration, 0.95),
            (0.43 * duration, 0.42),
            (0.66 * duration, 0.82),
            (0.88 * duration, 0.30),
            (duration, 0.0),
        ],
        sample_rate,
        gesture_size,
    )
    timing_rng = component_rng(seed, _PAPER_NAMESPACE, 1, sample_rate)
    outputs: list[tuple[FloatAudio, int]] = []
    spacing = round(duration * 0.82 * sample_rate)
    start = 0
    for index in range(pages):
        gesture = render_thin_material_source(
            preset=preset,
            activity=activity,
            sample_rate=sample_rate,
            seed=seed,
            event_index=index,
            amplitude=0.19 if speed == "normal" else 0.17,
        )
        if index:
            start += round(spacing * float(timing_rng.uniform(0.98, 1.02)))
        outputs.append((gesture, start))
    size = max(start + samples.size for samples, start in outputs)
    result = np.zeros(size, dtype=np.float32)
    for samples, start in outputs:
        mix_at(result, samples, start)
    return result


def render_paper_handle(
    *, sample_rate: int, duration: float, intensity: str, seed: int
) -> FloatAudio:
    preset = generate_thin_material_preset(seed=seed, sample_rate=sample_rate)
    size = max(1, round(duration * sample_rate))
    control_rng = component_rng(seed, _PAPER_NAMESPACE, 2, sample_rate)
    control_count = max(3, round(duration * 5.0))
    control_positions = np.linspace(0.0, max(0, size - 1), control_count)
    control_values = control_rng.uniform(0.32, 1.0, control_count)
    activity = np.interp(np.arange(size), control_positions, control_values).astype(np.float32)
    fade = envelope_from_points(
        [(0.0, 0.0), (min(0.08, duration * 0.2), 1.0), (max(0.09, duration - 0.10), 0.88), (duration, 0.0)],
        sample_rate,
        size,
    )
    activity *= fade
    amplitude = {"gentle": 0.12, "normal": 0.18, "rough": 0.26}[intensity]
    return render_thin_material_source(
        preset=preset,
        activity=activity,
        sample_rate=sample_rate,
        seed=seed,
        event_index=3,
        amplitude=amplitude,
    )

"""Seeded finite event populations for ambient sound composition."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .rng import component_rng

_STOCHASTIC_NAMESPACE = 0x53544F43


@dataclass(frozen=True, slots=True)
class StochasticEvent:
    """One bounded event sampled from a deterministic Poisson population."""

    start_sample: int
    duration_samples: int
    level: float
    event_index: int


def sample_stochastic_events(
    *,
    duration_s: float,
    sample_rate: int,
    rate_hz: float,
    seed: int,
    component_id: int,
    event_index: int = 0,
    event_duration_s: tuple[float, float] = (0.01, 0.08),
) -> tuple[StochasticEvent, ...]:
    """Sample seeded exponentially spaced events, bounded to a finite time span."""
    if not math.isfinite(duration_s) or duration_s <= 0.0:
        raise ValueError("duration_s must be finite and positive")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if not math.isfinite(rate_hz) or rate_hz < 0.0:
        raise ValueError("rate_hz must be finite and non-negative")
    minimum, maximum = event_duration_s
    if (
        not math.isfinite(minimum)
        or not math.isfinite(maximum)
        or minimum <= 0.0
        or maximum < minimum
    ):
        raise ValueError("event_duration_s must be a finite positive increasing pair")
    if rate_hz == 0.0:
        return ()

    rng = component_rng(seed, _STOCHASTIC_NAMESPACE, component_id, event_index)
    size = round(duration_s * sample_rate)
    time_s = 0.0
    events: list[StochasticEvent] = []
    while True:
        time_s += float(rng.exponential(1.0 / rate_hz))
        if time_s >= duration_s:
            break
        start_sample = round(time_s * sample_rate)
        if start_sample >= size:
            continue
        duration_samples = min(
            max(1, round(float(rng.uniform(minimum, maximum)) * sample_rate)),
            size - start_sample,
        )
        events.append(
            StochasticEvent(
                start_sample=start_sample,
                duration_samples=duration_samples,
                level=float(rng.uniform(0.45, 1.0)),
                event_index=len(events),
            )
        )
    return tuple(event for event in events if event.start_sample < size)

"""Stable independent random substreams for physical event components."""

from __future__ import annotations

from enum import IntEnum
from numbers import Integral

import numpy as np


class RandomStream(IntEnum):
    MOTION = 1
    CONTACT = 2
    ROUGHNESS = 3
    PARTICLES = 4
    POSITION = 5
    VARIATION = 6


def event_rng(seed: int, event_index: int, stream: RandomStream) -> np.random.Generator:
    """Create a deterministic stream unaffected by draws in other components."""
    if event_index < 0:
        raise ValueError("event_index must be non-negative")
    return np.random.default_rng(np.random.SeedSequence([int(seed), int(event_index), int(stream)]))


def component_rng(
    seed: int,
    namespace_id: int,
    component_id: int,
    event_index: int = 0,
) -> np.random.Generator:
    """Create a stable substream from explicit integer component identities."""
    values = (seed, namespace_id, component_id, event_index)
    names = ("seed", "namespace_id", "component_id", "event_index")
    integers: list[int] = []
    for name, value in zip(names, values, strict=True):
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(f"{name} must be an integer")
        integer = int(value)
        if integer < 0:
            raise ValueError(f"{name} must be non-negative")
        integers.append(integer)
    return np.random.default_rng(np.random.SeedSequence(integers))

"""Stable independent random substreams for physical event components."""

from __future__ import annotations

from enum import IntEnum

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

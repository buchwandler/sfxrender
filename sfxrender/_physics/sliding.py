"""Motion-coupled sliding sources built on the shared friction mechanism."""

from __future__ import annotations

from ..types import FloatAudio
from .friction import friction
from .models import FrictionProfile, ModalBody, MotionCurve
from .rng import component_rng

_SLIDING_NAMESPACE = 0x534C4944


def render_sliding_source(
    *,
    motion: MotionCurve,
    profile: FrictionProfile,
    sample_rate: int,
    seed: int,
    event_index: int = 0,
    body: ModalBody | None = None,
    activity: FloatAudio | None = None,
) -> FloatAudio:
    """Render a seeded friction bed and stick-slip motion over an optional body."""
    return friction(
        motion=motion,
        profile=profile,
        body=body,
        sample_rate=sample_rate,
        rng=component_rng(seed, _SLIDING_NAMESPACE, 1, event_index),
        activity=activity,
    )

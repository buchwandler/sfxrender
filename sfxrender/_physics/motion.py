"""Reusable normalized motion curves for perceptual physical interactions."""

from __future__ import annotations

import math

import numpy as np

from .models import MotionCurve


def eased_motion(
    *,
    duration_s: float,
    sample_rate: int,
    acceleration_shape: float,
    deceleration_shape: float,
) -> MotionCurve:
    """Return a smooth 0-to-1 motion and its finite-difference derivatives."""
    if not math.isfinite(duration_s) or duration_s <= 0.0:
        raise ValueError("duration_s must be finite and positive")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if (
        not math.isfinite(acceleration_shape)
        or acceleration_shape <= 0.0
        or not math.isfinite(deceleration_shape)
        or deceleration_shape <= 0.0
    ):
        raise ValueError("acceleration and deceleration shapes must be finite and positive")

    size = max(2, round(duration_s * sample_rate))
    time = np.linspace(0.0, 1.0, size, dtype=np.float64)
    accelerating = np.power(time, acceleration_shape)
    decelerating = np.power(1.0 - time, deceleration_shape)
    denominator = accelerating + decelerating
    position = np.divide(
        accelerating,
        denominator,
        out=np.zeros_like(accelerating),
        where=denominator > 0.0,
    )
    position[0] = 0.0
    position[-1] = 1.0
    sample_period = duration_s / (size - 1)
    velocity = np.gradient(position, sample_period)
    acceleration = np.gradient(velocity, sample_period)
    return MotionCurve(
        position=np.asarray(position, dtype=np.float32),
        velocity=np.asarray(velocity, dtype=np.float32),
        acceleration=np.asarray(acceleration, dtype=np.float32),
    )

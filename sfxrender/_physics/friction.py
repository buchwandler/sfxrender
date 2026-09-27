"""Motion-dependent rough friction and nonlinear harmonic stick-slip sources."""

from __future__ import annotations

import math

import numpy as np

from .._dsp import bandpass_noise, one_pole_lowpass, unit_rms
from ..types import FloatAudio
from .models import FrictionProfile, ModalBody, MotionCurve
from .resonators import resonate


def smooth_random_curve(
    rng: np.random.Generator,
    size: int,
    sample_rate: int,
    *,
    cutoff_hz: float = 6.0,
) -> FloatAudio:
    """Create seeded, low-rate normalized random control points."""
    if size <= 0:
        return np.zeros(0, dtype=np.float32)
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if not math.isfinite(cutoff_hz) or cutoff_hz <= 0.0:
        raise ValueError("cutoff_hz must be finite and positive")
    safe_cutoff = min(cutoff_hz, sample_rate * 0.2)
    knot_step = max(1, round(sample_rate / safe_cutoff))
    knot_count = max(2, math.ceil((size - 1) / knot_step) + 1)
    knot_positions = np.linspace(0.0, max(0, size - 1), knot_count)
    knot_values = rng.normal(0.0, 1.0, knot_count)
    curve = np.interp(np.arange(size), knot_positions, knot_values).astype(np.float32)
    curve = one_pole_lowpass(curve, sample_rate, min(safe_cutoff * 2.5, sample_rate * 0.45))
    normalized = unit_rms(curve)
    return np.asarray(np.tanh(normalized * np.float32(0.9)), dtype=np.float32)


def _piecewise_state(
    rng: np.random.Generator, size: int, sample_rate: int, *, change_rate_hz: float
) -> FloatAudio:
    step = max(1, round(sample_rate / change_rate_hz))
    count = max(2, math.ceil(size / step) + 1)
    positions = np.minimum(np.arange(count) * step, max(0, size - 1))
    values = rng.uniform(-1.0, 1.0, count)
    return np.interp(np.arange(size), positions, values).astype(np.float32)


def _intermittent_gate(rng: np.random.Generator, size: int, sample_rate: int) -> FloatAudio:
    gate = np.zeros(size, dtype=np.float32)
    cursor = 0
    ramp_size = max(1, round(sample_rate * 0.012))
    while cursor < size:
        run = max(1, round(float(rng.uniform(0.055, 0.24)) * sample_rate))
        end = min(size, cursor + run)
        if rng.random() < 0.68:
            gate[cursor:end] = np.float32(rng.uniform(0.72, 1.0))
            ramp = min(ramp_size, end - cursor)
            if ramp > 1:
                gate[cursor : cursor + ramp] *= np.linspace(0.0, 1.0, ramp, dtype=np.float32)
                gate[end - ramp : end] *= np.linspace(1.0, 0.0, ramp, dtype=np.float32)
        cursor = end
    return gate


def _validate_friction_profile(profile: FrictionProfile) -> None:
    for value, name in (
        (profile.base_gain, "base_gain"),
        (profile.roughness, "roughness"),
        (profile.stick_strength, "stick_strength"),
        (profile.slip_strength, "slip_strength"),
        (profile.chaos_amount, "chaos_amount"),
    ):
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")
    if profile.roughness > 1.0 or profile.chaos_amount > 1.0:
        raise ValueError("roughness and chaos_amount must not exceed 1")
    for pair, name in (
        (profile.noise_band_hz, "noise_band_hz"),
        (profile.f0_hz, "f0_hz"),
        (profile.harmonic_rolloff, "harmonic_rolloff"),
    ):
        if len(pair) != 2 or not all(math.isfinite(value) for value in pair) or pair[1] < pair[0]:
            raise ValueError(f"{name} must be an ordered finite pair")
    if profile.noise_band_hz[0] <= 0.0 or profile.f0_hz[0] <= 0.0:
        raise ValueError("noise band and f0 frequencies must be positive")
    if profile.harmonic_rolloff[0] <= 0.0:
        raise ValueError("harmonic rolloff must be positive")


def stick_slip_source(
    *,
    motion: MotionCurve,
    profile: FrictionProfile,
    activity: FloatAudio,
    sample_rate: int,
    rng: np.random.Generator,
    pitch_scale: FloatAudio | None = None,
) -> FloatAudio:
    """Generate irregular harmonic creak gated by motion and local activity."""
    _validate_friction_profile(profile)
    size = motion.position.size
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if activity.ndim != 1 or activity.size != size:
        raise ValueError("activity must match the motion curve length")
    if pitch_scale is not None and (pitch_scale.ndim != 1 or pitch_scale.size != size):
        raise ValueError("pitch_scale must match the motion curve length")
    pitch_curve = (
        np.ones(size, dtype=np.float32) if pitch_scale is None else np.clip(pitch_scale, 0.25, 4.0)
    )
    if size == 0:
        return np.zeros(0, dtype=np.float32)

    velocity = np.abs(motion.velocity).astype(np.float32)
    peak_velocity = float(np.max(velocity)) if velocity.size else 0.0
    if peak_velocity <= 1e-12:
        return np.zeros(size, dtype=np.float32)
    velocity = np.clip(velocity / np.float32(peak_velocity), 0.0, 1.0)
    local_activity = np.clip(activity, 0.0, 1.0)
    wander = smooth_random_curve(rng, size, sample_rate, cutoff_hz=5.5)
    states = _piecewise_state(rng, size, sample_rate, change_rate_hz=2.7)
    base_frequency = float(rng.uniform(profile.f0_hz[0], profile.f0_hz[1]))
    frequency = base_frequency * (
        1.0
        + 0.12 * velocity
        + profile.chaos_amount * 0.12 * wander
        + (0.045 + 0.12 * profile.chaos_amount) * states
    )
    frequency *= pitch_curve
    frequency = np.clip(frequency, 25.0, sample_rate * 0.42).astype(np.float32)
    phase = np.cumsum(
        np.float32(2.0 * math.pi) * frequency / np.float32(sample_rate), dtype=np.float32
    )

    harmonic_count = int(rng.integers(4, 13))
    rolloff = float(rng.uniform(profile.harmonic_rolloff[0], profile.harmonic_rolloff[1]))
    harmonics = np.zeros(size, dtype=np.float32)
    maximum_frequency = float(np.max(frequency))
    for harmonic in range(1, harmonic_count + 1):
        if maximum_frequency * harmonic >= sample_rate * 0.47:
            break
        phase_offset = float(rng.uniform(-math.pi, math.pi))
        harmonics += np.sin(harmonic * phase + phase_offset).astype(np.float32) / np.float32(
            harmonic**rolloff
        )
    gate = _intermittent_gate(rng, size, sample_rate)
    envelope = local_activity * velocity * gate
    level = profile.stick_strength * (0.65 + 0.35 * profile.roughness)
    return np.asarray(harmonics * envelope * np.float32(level), dtype=np.float32)


def friction(
    *,
    motion: MotionCurve,
    profile: FrictionProfile,
    body: ModalBody | None,
    sample_rate: int,
    rng: np.random.Generator,
    activity: FloatAudio | None = None,
    pitch_scale: FloatAudio | None = None,
) -> FloatAudio:
    """Render a rough sliding bed plus intermittent stick-slip and resonance."""
    _validate_friction_profile(profile)
    size = motion.position.size
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if activity is None:
        peak = float(np.max(np.abs(motion.velocity))) if size else 0.0
        if peak <= 1e-12:
            activity_signal = np.zeros(size, dtype=np.float32)
        else:
            activity_signal = np.clip(np.abs(motion.velocity) / np.float32(peak), 0.0, 1.0)
    else:
        if activity.ndim != 1 or activity.size != size:
            raise ValueError("activity must match the motion curve length")
        activity_signal = np.clip(activity, 0.0, 1.0)
    if pitch_scale is not None and (pitch_scale.ndim != 1 or pitch_scale.size != size):
        raise ValueError("pitch_scale must match the motion curve length")
    if size == 0:
        return np.zeros(0, dtype=np.float32)

    low = min(max(1.0, profile.noise_band_hz[0]), sample_rate * 0.40)
    high = min(max(low + 1.0, profile.noise_band_hz[1]), sample_rate * 0.46)
    noise = bandpass_noise(rng, size, sample_rate, low, high)
    velocity = np.abs(motion.velocity)
    peak_velocity = float(np.max(velocity)) if size else 0.0
    if peak_velocity > 1e-12:
        velocity = np.clip(velocity / np.float32(peak_velocity), 0.0, 1.0)
    else:
        velocity = np.zeros(size, dtype=np.float32)
    rough_level = profile.roughness * profile.slip_strength
    noise_layer = noise * activity_signal * velocity * np.float32(0.18 * rough_level)
    tonal = stick_slip_source(
        motion=motion,
        profile=profile,
        activity=activity_signal,
        pitch_scale=pitch_scale,
        sample_rate=sample_rate,
        rng=np.random.default_rng(int(rng.integers(0, 2**32, dtype=np.uint32))),
    )
    result = np.asarray((noise_layer + tonal) * np.float32(profile.base_gain), dtype=np.float32)
    if body is not None:
        resonated = resonate(
            result,
            body=body,
            sample_rate=sample_rate,
            rng=np.random.default_rng(int(rng.integers(0, 2**32, dtype=np.uint32))),
            frequency_jitter=0.008,
            decay_jitter=0.06,
        )
        result = result + resonated * np.float32(0.38)
    return np.asarray(result, dtype=np.float32)

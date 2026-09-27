"""Motion-dependent rough friction and nonlinear harmonic stick-slip sources."""

from __future__ import annotations

import math
from dataclasses import dataclass

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


@dataclass(frozen=True, slots=True)
class LuGrePreset:
    """Macro friction coefficients in SI units for one loaded interface."""

    static_coefficient: float
    dynamic_coefficient: float
    stribeck_velocity_m_s: float
    bristle_stiffness_n_m: float
    bristle_damping_n_s_m: float = 0.0
    viscous_coefficient_n_s_m: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.dynamic_coefficient <= self.static_coefficient:
            raise ValueError("friction coefficients must satisfy 0 <= dynamic <= static")
        for name in ("stribeck_velocity_m_s", "bristle_stiffness_n_m"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        for name in (
            "static_coefficient",
            "dynamic_coefficient",
            "bristle_damping_n_s_m",
            "viscous_coefficient_n_s_m",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class FrictionTrace:
    force_n: FloatAudio
    bristle_state_m: FloatAudio
    power_w: FloatAudio
    release_energy_j: FloatAudio


def lugre_friction(
    relative_velocity_m_s: FloatAudio,
    normal_load_n: float | FloatAudio,
    preset: LuGrePreset,
    sample_rate: int,
    *,
    initial_state_m: float = 0.0,
) -> FrictionTrace:
    """Integrate a stable LuGre bristle state and return opposing friction force."""
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    velocity = np.asarray(relative_velocity_m_s, dtype=np.float64)
    if velocity.ndim != 1 or not np.all(np.isfinite(velocity)):
        raise ValueError("relative_velocity_m_s must be a finite one-dimensional signal")
    if isinstance(normal_load_n, np.ndarray):
        load = np.asarray(normal_load_n, dtype=np.float64)
    else:
        load = np.full(velocity.size, float(normal_load_n), dtype=np.float64)
    if load.shape != velocity.shape or not np.all(np.isfinite(load)) or np.any(load < 0.0):
        raise ValueError("normal_load_n must be finite, non-negative, and match velocity")
    if not math.isfinite(initial_state_m):
        raise ValueError("initial_state_m must be finite")
    forces = np.zeros(velocity.size, dtype=np.float64)
    states = np.zeros(velocity.size, dtype=np.float64)
    powers = np.zeros(velocity.size, dtype=np.float64)
    releases = np.zeros(velocity.size, dtype=np.float64)
    state = float(initial_state_m)
    dt = 1.0 / sample_rate
    state_limit_scale = preset.static_coefficient / preset.bristle_stiffness_n_m
    for index, speed in enumerate(velocity):
        previous = state
        abs_speed = abs(float(speed))
        mu = preset.dynamic_coefficient + (
            preset.static_coefficient - preset.dynamic_coefficient
        ) * math.exp(-((abs_speed / preset.stribeck_velocity_m_s) ** 2))
        limiting_force = mu * float(load[index])
        g_distance = max(limiting_force / preset.bristle_stiffness_n_m, 1e-12)
        rate = abs_speed / g_distance
        if rate <= 1e-12:
            state = previous
        else:
            decay = math.exp(-rate * dt)
            state = previous * decay + float(speed) * (-math.expm1(-rate * dt)) / rate
        state = min(
            max(state, -state_limit_scale * float(load[index])),
            state_limit_scale * float(load[index]),
        )
        state_rate = (state - previous) / dt
        raw = (
            preset.bristle_stiffness_n_m * state
            + preset.bristle_damping_n_s_m * state_rate
            + preset.viscous_coefficient_n_s_m * float(speed)
        )
        if abs_speed <= 1e-12:
            force = -preset.bristle_stiffness_n_m * state
        else:
            opposing_magnitude = max(0.0, math.copysign(1.0, float(speed)) * raw)
            force = -math.copysign(opposing_magnitude, float(speed))
        forces[index] = force
        states[index] = state
        powers[index] = max(0.0, -force * float(speed))
        releases[index] = max(
            0.0,
            0.5 * preset.bristle_stiffness_n_m * (previous * previous - state * state),
        )
    return FrictionTrace(
        force_n=np.asarray(forces, dtype=np.float32),
        bristle_state_m=np.asarray(states, dtype=np.float32),
        power_w=np.asarray(powers, dtype=np.float32),
        release_energy_j=np.asarray(releases, dtype=np.float32),
    )


@dataclass(frozen=True, slots=True)
class RoughnessProfile:
    spatial_sample_rate_per_m: int
    heights_m: FloatAudio
    correlation_length_m: float

    def __post_init__(self) -> None:
        if (
            self.spatial_sample_rate_per_m <= 0
            or self.heights_m.ndim != 1
            or self.heights_m.size < 2
        ):
            raise ValueError("roughness profile needs a positive rate and at least two samples")
        if not np.all(np.isfinite(self.heights_m)):
            raise ValueError("roughness heights must be finite")
        if not math.isfinite(self.correlation_length_m) or self.correlation_length_m <= 0.0:
            raise ValueError("correlation_length_m must be finite and positive")


def make_roughness_profile(
    *,
    roughness_rms_m: float,
    correlation_length_m: float,
    seed: int,
    length_m: float = 0.15,
    spatial_sample_rate_per_m: int | None = None,
    spectral_slope: float = 2.0,
) -> RoughnessProfile:
    """Generate a compact deterministic band-limited spatial surface track."""
    for value, name in (
        (roughness_rms_m, "roughness_rms_m"),
        (correlation_length_m, "correlation_length_m"),
        (length_m, "length_m"),
        (spectral_slope, "spectral_slope"),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    rate = spatial_sample_rate_per_m or max(1_000, math.ceil(8.0 / correlation_length_m))
    if rate <= 0 or seed < 0:
        raise ValueError("spatial sample rate must be positive and seed non-negative")
    size = max(2, math.ceil(length_m * rate))
    rng = np.random.default_rng(seed)
    white = rng.normal(size=size)
    spectrum = np.fft.rfft(white)
    spatial_frequency = np.fft.rfftfreq(size, d=1.0 / rate)
    shaping = (1.0 + (spatial_frequency * correlation_length_m) ** 2) ** (-spectral_slope / 4.0)
    shaping[0] = 0.0
    heights = np.fft.irfft(spectrum * shaping, n=size)
    standard_deviation = float(np.std(heights))
    if standard_deviation > 0.0:
        heights *= roughness_rms_m / standard_deviation
    return RoughnessProfile(
        spatial_sample_rate_per_m=rate,
        heights_m=np.asarray(heights, dtype=np.float32),
        correlation_length_m=correlation_length_m,
    )


def roughness_velocity(
    profile: RoughnessProfile,
    position_m: FloatAudio,
    sample_rate: int,
) -> FloatAudio:
    """Sample a fixed spatial profile along a trajectory and return dr/dt."""
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    positions = np.asarray(position_m, dtype=np.float64)
    if positions.ndim != 1 or not np.all(np.isfinite(positions)):
        raise ValueError("position_m must be a finite one-dimensional signal")
    coordinates = (
        np.arange(profile.heights_m.size, dtype=np.float64) / profile.spatial_sample_rate_per_m
    )
    heights = np.interp(positions, coordinates, profile.heights_m.astype(np.float64))
    if heights.size < 2:
        return np.zeros(heights.size, dtype=np.float32)
    result = np.gradient(heights) * sample_rate
    return np.asarray(result, dtype=np.float32)

"""Tests for reusable impact and motion-driven friction interactions."""

from __future__ import annotations

import numpy as np

from sfxrender._physics.friction import friction, smooth_random_curve, stick_slip_source
from sfxrender._physics.impacts import impact
from sfxrender._physics.models import ContactProfile, FrictionProfile, MotionCurve
from sfxrender._physics.motion import eased_motion

_CONTACT = ContactProfile(hardness=0.55, brightness=0.48, roughness=0.4, contact_gain=0.7)
_FRICTION = FrictionProfile(
    base_gain=0.5,
    roughness=0.55,
    noise_band_hz=(180.0, 3_800.0),
    stick_strength=0.7,
    slip_strength=0.6,
    f0_hz=(115.0, 210.0),
    harmonic_rolloff=(1.1, 1.8),
    chaos_amount=0.65,
)


def _rms(samples: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))


def test_impact_is_seeded_and_force_changes_excitation_energy() -> None:
    low = impact(
        contact=_CONTACT,
        force=0.2,
        duration_s=0.18,
        body=None,
        sample_rate=12_000,
        rng=np.random.default_rng(7),
    )
    repeated = impact(
        contact=_CONTACT,
        force=0.2,
        duration_s=0.18,
        body=None,
        sample_rate=12_000,
        rng=np.random.default_rng(7),
    )
    high = impact(
        contact=_CONTACT,
        force=0.9,
        duration_s=0.18,
        body=None,
        sample_rate=12_000,
        rng=np.random.default_rng(7),
    )
    np.testing.assert_array_equal(low, repeated)
    assert low.dtype == np.float32
    assert np.all(np.isfinite(low))
    assert _rms(high) > _rms(low) * 3.0


def test_smooth_random_control_is_seeded_low_frequency_float32() -> None:
    first = smooth_random_curve(np.random.default_rng(23), 8_000, 8_000, cutoff_hz=5.0)
    repeated = smooth_random_curve(np.random.default_rng(23), 8_000, 8_000, cutoff_hz=5.0)
    changed = smooth_random_curve(np.random.default_rng(24), 8_000, 8_000, cutoff_hz=5.0)
    np.testing.assert_array_equal(first, repeated)
    assert not np.array_equal(first, changed)
    assert first.dtype == np.float32
    assert np.all(np.isfinite(first))
    assert float(np.max(np.abs(np.diff(first)))) < 0.1


def test_stick_slip_is_seeded_harmonic_and_motion_gated() -> None:
    motion = eased_motion(
        duration_s=1.2,
        sample_rate=8_000,
        acceleration_shape=1.3,
        deceleration_shape=1.7,
    )
    activity = np.zeros(motion.position.size, dtype=np.float32)
    activity[800:2_300] = 0.85
    activity[4_500:6_800] = 1.0
    first = stick_slip_source(
        motion=motion,
        profile=_FRICTION,
        activity=activity,
        sample_rate=8_000,
        rng=np.random.default_rng(81),
    )
    repeated = stick_slip_source(
        motion=motion,
        profile=_FRICTION,
        activity=activity,
        sample_rate=8_000,
        rng=np.random.default_rng(81),
    )
    np.testing.assert_array_equal(first, repeated)
    assert first.dtype == np.float32
    assert np.all(np.isfinite(first))
    assert np.any(first[800:2_300] != 0.0)
    assert np.any(first[4_500:6_800] != 0.0)
    assert not np.any(first[:800])
    assert not np.any(first[2_300:4_500])
    frame = first[800:2_300].reshape(-1, 100)
    frame_rms = np.sqrt(np.mean(np.square(frame, dtype=np.float64), axis=1))
    assert float(np.std(frame_rms)) > 0.0


def test_friction_layer_is_seeded_and_motion_dependent() -> None:
    motion = eased_motion(
        duration_s=0.8,
        sample_rate=8_000,
        acceleration_shape=1.0,
        deceleration_shape=1.0,
    )
    first = friction(
        motion=motion,
        profile=_FRICTION,
        body=None,
        sample_rate=8_000,
        rng=np.random.default_rng(91),
    )
    repeated = friction(
        motion=motion,
        profile=_FRICTION,
        body=None,
        sample_rate=8_000,
        rng=np.random.default_rng(91),
    )
    np.testing.assert_array_equal(first, repeated)
    assert np.all(np.isfinite(first))
    assert _rms(first) > 0.0
    assert first.dtype == np.float32
    stationary = MotionCurve(
        position=np.zeros(2_000, dtype=np.float32),
        velocity=np.zeros(2_000, dtype=np.float32),
        acceleration=np.zeros(2_000, dtype=np.float32),
    )
    silent = friction(
        motion=stationary,
        profile=_FRICTION,
        body=None,
        sample_rate=8_000,
        rng=np.random.default_rng(92),
    )
    assert _rms(silent) == 0.0

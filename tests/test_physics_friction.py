"""Tests for macro LuGre friction and deterministic spatial roughness."""

from __future__ import annotations

import numpy as np
import pytest

from sfxrender._physics.friction import (
    LuGrePreset,
    lugre_friction,
    make_roughness_profile,
    roughness_velocity,
)


def _preset() -> LuGrePreset:
    return LuGrePreset(
        static_coefficient=0.8,
        dynamic_coefficient=0.5,
        stribeck_velocity_m_s=0.04,
        bristle_stiffness_n_m=1_000.0,
        bristle_damping_n_s_m=0.05,
        viscous_coefficient_n_s_m=0.02,
    )


def test_lugre_force_opposes_sliding_and_approaches_kinetic_friction() -> None:
    velocity = np.full(8_000, 0.25, dtype=np.float32)
    forward = lugre_friction(velocity, 10.0, _preset(), 8_000)
    reverse = lugre_friction(-velocity, 10.0, _preset(), 8_000)
    assert np.all(forward.force_n <= 0.0)
    assert np.all(reverse.force_n >= 0.0)
    assert np.median(np.abs(forward.force_n[-1_000:])) == pytest.approx(5.0, rel=0.08)
    assert np.all(forward.power_w >= 0.0)
    assert float(np.sum(forward.power_w)) > 0.0


def test_lugre_retains_static_bristle_force_and_nonnegative_dissipation() -> None:
    velocity = np.zeros(100, dtype=np.float32)
    preset = _preset()
    state = preset.static_coefficient * 10.0 / preset.bristle_stiffness_n_m * 0.5
    trace = lugre_friction(velocity, 10.0, preset, 1_000, initial_state_m=state)
    assert np.all(trace.force_n < 0.0)
    assert np.all(trace.power_w == 0.0)
    assert np.all(np.isfinite(trace.bristle_state_m))
    sliding = lugre_friction(np.full(2_000, 0.1, dtype=np.float32), 10.0, preset, 1_000)
    assert np.all(sliding.force_n * 0.1 <= 0.0)
    assert np.all(sliding.power_w >= 0.0)
    assert np.all(sliding.release_energy_j >= 0.0)


def test_lugre_accepts_audio_rate_load_control_and_rejects_bad_shapes() -> None:
    velocity = np.full(200, 0.08, dtype=np.float32)
    load = np.linspace(1.0, 15.0, velocity.size, dtype=np.float32)
    trace = lugre_friction(velocity, load, _preset(), 4_000)
    assert trace.force_n.shape == velocity.shape
    assert np.all(np.isfinite(trace.force_n))
    with pytest.raises(ValueError, match="match velocity"):
        lugre_friction(velocity, np.ones(20, dtype=np.float32), _preset(), 4_000)


def test_spatial_roughness_is_seeded_bandlimited_and_traversal_dependent() -> None:
    first = make_roughness_profile(
        roughness_rms_m=2e-6,
        correlation_length_m=7e-4,
        seed=33,
        length_m=0.12,
    )
    repeated = make_roughness_profile(
        roughness_rms_m=2e-6,
        correlation_length_m=7e-4,
        seed=33,
        length_m=0.12,
    )
    changed = make_roughness_profile(
        roughness_rms_m=2e-6,
        correlation_length_m=7e-4,
        seed=34,
        length_m=0.12,
    )
    np.testing.assert_array_equal(first.heights_m, repeated.heights_m)
    assert not np.array_equal(first.heights_m, changed.heights_m)
    assert float(np.std(first.heights_m)) == pytest.approx(2e-6, rel=1e-5)
    forward = roughness_velocity(first, np.linspace(0.0, 0.03, 1_000, dtype=np.float32), 8_000)
    reverse = roughness_velocity(first, np.linspace(0.03, 0.0, 1_000, dtype=np.float32), 8_000)
    assert forward.dtype == np.float32
    assert np.all(np.isfinite(forward))
    assert not np.array_equal(forward, reverse)

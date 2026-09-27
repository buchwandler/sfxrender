"""Regression tests for the physical door-motion model."""

from __future__ import annotations

import numpy as np
import pytest

from sfxrender._doors import generate_door_model, render_open
from sfxrender._physics.motion import minimum_jerk_motion


def test_minimum_jerk_motion_has_smooth_endpoints_and_unit_travel() -> None:
    duration_s = 1.7
    motion = minimum_jerk_motion(duration_s=duration_s, sample_rate=2_000)
    assert motion.position[0] == 0.0
    assert motion.position[-1] == 1.0
    assert np.all(np.diff(motion.position) >= 0.0)
    assert motion.velocity[0] == motion.velocity[-1] == 0.0
    assert motion.acceleration[0] == motion.acceleration[-1] == 0.0
    assert float(np.max(motion.velocity)) == pytest.approx(1.875 / duration_s, rel=2e-4)


def test_door_identity_carries_plate_modes_hinge_emitters_and_inertia() -> None:
    wood = generate_door_model(material="wood", seed=17, sample_rate=24_000)
    metal = generate_door_model(material="metal", seed=17, sample_rate=24_000)
    assert wood.mass_kg == 27.0
    assert wood.inertia_kg_m2 == pytest.approx(wood.mass_kg * wood.width_m**2 / 3.0)
    assert wood.width_m != metal.width_m
    assert len(wood.panel_modes.modes) >= 8
    assert len(wood.hinge_mode_sets) == 3
    assert all(modes.modes for modes in wood.hinge_mode_sets)
    assert wood.hinge_mode_sets[0].modes[0].frequency_hz == pytest.approx(
        wood.hinge_mode_sets[1].modes[0].frequency_hz
    )
    assert not np.array_equal(
        [mode.input_gain for mode in wood.hinge_mode_sets[0].modes],
        [mode.input_gain for mode in wood.hinge_mode_sets[1].modes],
    )


def test_door_friction_audio_is_deterministic_and_creak_controls_energy() -> None:
    model = generate_door_model(material="wood", seed=42, sample_rate=12_000)
    quiet = render_open(model=model, speed="normal", creak=0.0, force=0.7, sample_rate=12_000)
    repeated = render_open(model=model, speed="normal", creak=0.0, force=0.7, sample_rate=12_000)
    rough = render_open(model=model, speed="normal", creak=1.0, force=0.7, sample_rate=12_000)
    np.testing.assert_array_equal(quiet, repeated)
    movement = slice(int(0.2 * 12_000), int(1.2 * 12_000))
    quiet_rms = float(np.sqrt(np.mean(quiet[movement] ** 2)))
    rough_rms = float(np.sqrt(np.mean(rough[movement] ** 2)))
    assert rough_rms > quiet_rms * 8.0
    assert np.all(np.isfinite(rough))

"""Regression tests for force-driven geometry-aware knock rendering."""

from __future__ import annotations

import numpy as np

from sfxrender import SFXRenderer
from sfxrender._physics.geometry import rectangular_plate_modes
from sfxrender._physics.presets import KNOCK_IMPACTORS, KNOCK_OBJECTS, OBJECT_PRESETS


def test_knock_strike_position_changes_participation_not_object_frequencies() -> None:
    oak_door = OBJECT_PRESETS[KNOCK_OBJECTS["oak"]]
    center = rectangular_plate_modes(
        oak_door.geometry,
        oak_door.material,
        sample_rate=24_000,
        max_modes=32,
        contact_x=0.5,
        contact_y=0.5,
    )
    edge = rectangular_plate_modes(
        oak_door.geometry,
        oak_door.material,
        sample_rate=24_000,
        max_modes=32,
        contact_x=0.76,
        contact_y=0.37,
    )
    np.testing.assert_array_equal(
        [mode.frequency_hz for mode in center.modes],
        [mode.frequency_hz for mode in edge.modes],
    )
    assert not np.array_equal(
        [mode.input_gain for mode in center.modes],
        [mode.input_gain for mode in edge.modes],
    )


def test_impactor_presets_have_distinct_mechanical_stiffness_and_mass() -> None:
    assert KNOCK_IMPACTORS["fingertip"].stiffness < KNOCK_IMPACTORS["knuckle"].stiffness
    assert KNOCK_IMPACTORS["knuckle"].stiffness < KNOCK_IMPACTORS["metal_object"].stiffness
    assert KNOCK_IMPACTORS["wooden_object"].effective_mass_kg > (
        KNOCK_IMPACTORS["knuckle"].effective_mass_kg
    )


def test_public_knock_uri_and_impactor_render_are_deterministic() -> None:
    renderer = SFXRenderer(sample_rate=16_000)
    uri = "sfx:impact.knock?material=oak&impactor=metal_object&force=0.7&seed=42"
    first = renderer.render_uri(uri)
    repeated = renderer.render_uri(uri)
    np.testing.assert_array_equal(first.samples, repeated.samples)
    assert first.samples.ndim == 1
    assert first.samples.dtype == np.float32
    assert np.all(np.isfinite(first.samples))
    assert float(np.max(np.abs(first.samples))) <= 1.0

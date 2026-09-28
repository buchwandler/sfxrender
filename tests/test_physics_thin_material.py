from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from sfxrender._physics.thin_material import (
    generate_thin_material_preset,
    render_thin_material_source,
)


@pytest.mark.parametrize("sample_rate", [8_000, 16_000, 24_000, 48_000])
def test_thin_material_source_is_seeded_finite_and_tailed(sample_rate: int) -> None:
    activity = np.full(round(0.7 * sample_rate), 0.65, dtype=np.float32)
    preset = generate_thin_material_preset(seed=42, sample_rate=sample_rate)
    first = render_thin_material_source(
        preset=preset,
        activity=activity,
        sample_rate=sample_rate,
        seed=42,
    )
    repeated = render_thin_material_source(
        preset=preset,
        activity=activity,
        sample_rate=sample_rate,
        seed=42,
    )
    changed = render_thin_material_source(
        preset=preset,
        activity=activity,
        sample_rate=sample_rate,
        seed=43,
    )

    np.testing.assert_array_equal(first, repeated)
    assert not np.array_equal(first, changed)
    assert first.size == activity.size + round(preset.tail_s * sample_rate)
    assert first.dtype == np.float32
    assert np.isfinite(first).all()
    assert float(np.max(np.abs(first))) > 1e-5
    assert float(np.max(np.abs(first))) < 1.0


def test_thin_material_activity_controls_continuous_friction() -> None:
    sample_rate = 24_000
    preset = generate_thin_material_preset(seed=7, sample_rate=sample_rate)
    active = np.full(round(0.4 * sample_rate), 0.8, dtype=np.float32)
    silent = np.zeros_like(active)
    source = render_thin_material_source(
        preset=preset,
        activity=active,
        sample_rate=sample_rate,
        seed=7,
    )
    quiet = render_thin_material_source(
        preset=preset,
        activity=silent,
        sample_rate=sample_rate,
        seed=7,
    )

    assert np.any(source)
    assert not np.any(quiet)


def test_thin_material_has_separate_flutter_events() -> None:
    sample_rate = 24_000
    preset = generate_thin_material_preset(seed=9, sample_rate=sample_rate)
    activity = np.full(round(0.5 * sample_rate), 0.7, dtype=np.float32)
    with_events = render_thin_material_source(
        preset=preset,
        activity=activity,
        sample_rate=sample_rate,
        seed=9,
    )
    continuous_only = render_thin_material_source(
        preset=replace(preset, event_rate_hz=0.0),
        activity=activity,
        sample_rate=sample_rate,
        seed=9,
    )
    assert not np.array_equal(with_events, continuous_only)

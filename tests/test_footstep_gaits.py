from __future__ import annotations

import numpy as np
import pytest

from sfxrender import InvalidEffectParameterError, SFXRenderer

_GAIT_URIS = (
    "sfx:footsteps.run?surface=wood&footwear=boots&count=3&force=0.58&interval=0.32&seed=17",
    "sfx:footsteps.stairs?surface=wood&footwear=boots&direction=up&count=3&force=0.58&seed=17",
)


def _rms(samples: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(samples.astype(np.float64)))))


@pytest.mark.parametrize("sample_rate", [8_000, 16_000, 24_000, 48_000])
@pytest.mark.parametrize("uri", _GAIT_URIS)
def test_new_footstep_gaits_obey_seeded_pcm_contract(sample_rate: int, uri: str) -> None:
    renderer = SFXRenderer(sample_rate=sample_rate)
    first = renderer.render_uri(uri)
    repeated = renderer.render_uri(uri)
    changed = renderer.render_uri(uri.replace("seed=17", "seed=18"))

    np.testing.assert_array_equal(first.samples, repeated.samples)
    assert not np.array_equal(first.samples, changed.samples)
    assert first.sample_rate == sample_rate
    assert first.samples.ndim == 1
    assert first.samples.dtype == np.float32
    assert first.samples.size > 0
    assert np.isfinite(first.samples).all()
    assert float(np.max(np.abs(first.samples))) <= 1.0
    assert _rms(first.samples) > 1e-5


def test_run_uses_a_shorter_stride_than_walking() -> None:
    renderer = SFXRenderer(sample_rate=24_000)
    walk = renderer.render_uri(
        "sfx:footsteps.walk?surface=wood&footwear=boots&count=6&interval=0.52&seed=42"
    )
    run = renderer.render_uri(
        "sfx:footsteps.run?surface=wood&footwear=boots&count=6&interval=0.32&seed=42"
    )
    assert run.duration < walk.duration


@pytest.mark.parametrize(
    ("effect", "parameters"),
    [
        ("footsteps.run", "surface=wood&footwear=boots&count=1&interval=0.32"),
        ("footsteps.stairs", "surface=wood&footwear=boots&direction=up&count=1"),
    ],
)
def test_gait_force_has_a_monotonic_level_effect(effect: str, parameters: str) -> None:
    renderer = SFXRenderer(sample_rate=24_000)
    gentle = renderer.render_uri(f"sfx:{effect}?{parameters}&force=0.2&seed=42")
    firm = renderer.render_uri(f"sfx:{effect}?{parameters}&force=0.65&seed=42")
    assert _rms(firm.samples) > _rms(gentle.samples)


def test_stair_direction_and_tread_material_change_the_response() -> None:
    renderer = SFXRenderer(sample_rate=24_000)
    up = renderer.render_uri("sfx:footsteps.stairs?surface=wood&direction=up&count=3&seed=42")
    down = renderer.render_uri("sfx:footsteps.stairs?surface=wood&direction=down&count=3&seed=42")
    stone = renderer.render_uri("sfx:footsteps.stairs?surface=stone&direction=up&count=3&seed=42")
    assert not np.array_equal(up.samples, down.samples)
    assert not np.array_equal(up.samples, stone.samples)


def test_stairs_limit_surfaces_to_existing_tread_models() -> None:
    with pytest.raises(
        InvalidEffectParameterError,
        match="Invalid parameter 'surface' for effect 'footsteps.stairs'",
    ):
        SFXRenderer().render_uri("sfx:footsteps.stairs?surface=metal")

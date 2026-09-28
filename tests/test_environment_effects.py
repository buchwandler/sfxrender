from __future__ import annotations

import numpy as np
import pytest

from sfxrender import SFXRenderer

_ENVIRONMENT_URIS = (
    "sfx:wind?intensity=light&texture=smooth&duration=1.0&seed=42",
    "sfx:transition.whoosh?style=soft&direction=rise&duration=0.5&seed=42",
    "sfx:rain?intensity=light&surface=ground&duration=1.0&seed=42",
    "sfx:fire.crackle?activity=quiet&duration=2.0&seed=42",
    "sfx:birds.ambience?activity=sparse&duration=2.0&seed=42",
    "sfx:crickets.ambience?activity=sparse&duration=2.0&seed=42",
)


def _rms(samples: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(samples.astype(np.float64)))))


@pytest.mark.parametrize("sample_rate", [8_000, 16_000, 24_000, 48_000])
@pytest.mark.parametrize("uri", _ENVIRONMENT_URIS)
def test_environment_effects_are_seeded_finite_pcm(sample_rate: int, uri: str) -> None:
    renderer = SFXRenderer(sample_rate=sample_rate)
    first = renderer.render_uri(uri)
    repeated = renderer.render_uri(uri)
    changed = renderer.render_uri(uri.replace("seed=42", "seed=43"))

    np.testing.assert_array_equal(first.samples, repeated.samples)
    assert not np.array_equal(first.samples, changed.samples)
    assert first.sample_rate == sample_rate
    assert first.samples.dtype == np.float32
    assert first.samples.ndim == 1
    assert first.samples.size > 0
    assert first.samples.size / sample_rate == pytest.approx(first.duration)
    assert np.isfinite(first.samples).all()
    assert float(np.max(np.abs(first.samples))) <= 1.0
    assert 1e-5 < _rms(first.samples) < 0.2


def test_environment_parameters_change_their_semantic_texture() -> None:
    renderer = SFXRenderer(sample_rate=24_000)
    pairs = (
        (
            "sfx:wind?intensity=light&texture=smooth&duration=3.0&seed=42",
            "sfx:wind?intensity=strong&texture=leafy&duration=3.0&seed=42",
        ),
        (
            "sfx:transition.whoosh?style=soft&direction=rise&duration=1.2&seed=42",
            "sfx:transition.whoosh?style=forceful&direction=fall&duration=1.2&seed=42",
        ),
        (
            "sfx:rain?intensity=light&surface=ground&duration=2.0&seed=42",
            "sfx:rain?intensity=heavy&surface=roof&duration=2.0&seed=42",
        ),
        (
            "sfx:fire.crackle?activity=quiet&duration=6.0&seed=42",
            "sfx:fire.crackle?activity=active&duration=6.0&seed=42",
        ),
        (
            "sfx:birds.ambience?activity=sparse&duration=8.0&seed=42",
            "sfx:birds.ambience?activity=busy&duration=8.0&seed=42",
        ),
        (
            "sfx:crickets.ambience?activity=sparse&duration=8.0&seed=42",
            "sfx:crickets.ambience?activity=busy&duration=8.0&seed=42",
        ),
    )
    for first_uri, second_uri in pairs:
        first = renderer.render_uri(first_uri)
        second = renderer.render_uri(second_uri)
        assert not np.array_equal(first.samples, second.samples)

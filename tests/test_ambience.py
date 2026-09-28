from __future__ import annotations

import numpy as np
import pytest

from sfxrender import SFXRenderer

_AMBIENCE_URIS = (
    "sfx:room_tone?character=quiet&duration=0.8&seed=42",
    "sfx:office.ambience?activity=quiet&duration=2.0&seed=42",
    "sfx:city.ambience?activity=calm&duration=4.0&seed=42",
)


def _rms(samples: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(samples.astype(np.float64)))))


@pytest.mark.parametrize("sample_rate", [8_000, 16_000, 24_000, 48_000])
@pytest.mark.parametrize("uri", _AMBIENCE_URIS)
def test_ambience_is_seeded_finite_pcm(sample_rate: int, uri: str) -> None:
    renderer = SFXRenderer(sample_rate=sample_rate)
    first = renderer.render_uri(uri)
    repeated = renderer.render_uri(uri)
    changed = renderer.render_uri(uri.replace("seed=42", "seed=43"))

    np.testing.assert_array_equal(first.samples, repeated.samples)
    assert not np.array_equal(first.samples, changed.samples)
    assert first.sample_rate == sample_rate
    assert first.samples.dtype == np.float32
    assert first.samples.size > 0
    assert first.samples.size / sample_rate == pytest.approx(first.duration)
    assert np.isfinite(first.samples).all()
    assert float(np.max(np.abs(first.samples))) <= 1.0
    assert 1e-5 < _rms(first.samples) < 0.1


def test_room_tone_profiles_and_activity_levels_change_the_composition() -> None:
    renderer = SFXRenderer(sample_rate=24_000)
    quiet = renderer.render_uri("sfx:room_tone?character=quiet&duration=2.0&seed=42")
    electrical = renderer.render_uri("sfx:room_tone?character=electrical&duration=2.0&seed=42")
    office_quiet = renderer.render_uri("sfx:office.ambience?activity=quiet&duration=10.0&seed=42")
    office_busy = renderer.render_uri("sfx:office.ambience?activity=busy&duration=10.0&seed=42")
    city_calm = renderer.render_uri("sfx:city.ambience?activity=calm&duration=20.0&seed=42")
    city_busy = renderer.render_uri("sfx:city.ambience?activity=busy&duration=20.0&seed=42")

    assert not np.array_equal(quiet.samples, electrical.samples)
    assert not np.array_equal(office_quiet.samples, office_busy.samples)
    assert not np.array_equal(city_calm.samples, city_busy.samples)

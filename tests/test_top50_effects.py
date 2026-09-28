from __future__ import annotations

import numpy as np
import pytest

from sfxrender import SFXRenderer

_TOP50_URIS = (
    "sfx:keys.jingle?style=full&duration=0.5&seed=42",
    "sfx:lock.turn?style=key&force=light&duration=0.3&seed=42",
    "sfx:chair.move?surface=wood&effort=light&action=slide&duration=0.5&seed=42",
    "sfx:cloth.rustle?fabric=cotton&activity=light&duration=0.5&seed=42",
    "sfx:floor.creak?surface=wood&weight=light&duration=0.4&seed=42",
    "sfx:keyboard.typing?speed=steady&force=light&duration=1.0&seed=42",
    "sfx:clock.tick?style=wall&rate=normal&duration=1.0&seed=42",
    "sfx:alarm.ring?style=mechanical&pattern=intermittent&duration=1.0&seed=42",
    "sfx:elevator.arrive?size=small&chime=single&duration=1.0&seed=42",
    "sfx:water.pour?flow=steady&vessel=glass&duration=0.5&seed=42",
    "sfx:water.running?flow=steady&duration=1.0&seed=42",
    "sfx:crowd.murmur?density=sparse&duration=2.0&seed=42",
    "sfx:thunder?intensity=strong&distance=distant&duration=3.0&seed=42",
    "sfx:car.door?action=close&size=sedan&force=firm&duration=0.5&seed=42",
    "sfx:car.engine?action=idle&vehicle=compact&duration=1.0&seed=42",
    "sfx:car.passby?speed=slow&vehicle=compact&duration=2.0&seed=42",
)


def _rms(samples: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(samples.astype(np.float64)))))


@pytest.mark.parametrize("sample_rate", [8_000, 16_000, 24_000, 48_000])
@pytest.mark.parametrize("uri", _TOP50_URIS)
def test_remaining_top50_effects_are_seeded_finite_pcm(sample_rate: int, uri: str) -> None:
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
    assert np.isfinite(first.samples).all()
    assert float(np.max(np.abs(first.samples))) <= 1.0
    assert _rms(first.samples) > 1e-5


def test_remaining_top50_semantic_parameters_change_the_sound() -> None:
    renderer = SFXRenderer(sample_rate=24_000)
    pairs = (
        (
            "sfx:keys.jingle?style=light&duration=1.5&seed=42",
            "sfx:keys.jingle?style=full&duration=1.5&seed=42",
        ),
        (
            "sfx:lock.turn?style=key&force=light&duration=1.5&seed=42",
            "sfx:lock.turn?style=deadbolt&force=firm&duration=1.5&seed=42",
        ),
        (
            "sfx:chair.move?surface=wood&effort=light&duration=2.0&seed=42",
            "sfx:chair.move?surface=stone&effort=firm&duration=2.0&seed=42",
        ),
        (
            "sfx:cloth.rustle?fabric=cotton&activity=light&duration=3.0&seed=42",
            "sfx:cloth.rustle?fabric=silk&activity=active&duration=3.0&seed=42",
        ),
        (
            "sfx:floor.creak?surface=wood&weight=light&duration=1.3&seed=42",
            "sfx:floor.creak?surface=carpet&weight=heavy&duration=1.3&seed=42",
        ),
        (
            "sfx:keyboard.typing?speed=slow&force=light&duration=4.0&seed=42",
            "sfx:keyboard.typing?speed=fast&force=firm&duration=4.0&seed=42",
        ),
        (
            "sfx:clock.tick?style=wall&rate=slow&duration=4.0&seed=42",
            "sfx:clock.tick?style=mantel&rate=normal&duration=4.0&seed=42",
        ),
        (
            "sfx:alarm.ring?style=mechanical&pattern=intermittent&duration=3.0&seed=42",
            "sfx:alarm.ring?style=electronic&pattern=continuous&duration=3.0&seed=42",
        ),
        (
            "sfx:elevator.arrive?size=small&chime=single&duration=3.5&seed=42",
            "sfx:elevator.arrive?size=large&chime=double&duration=3.5&seed=42",
        ),
        (
            "sfx:water.pour?flow=trickle&vessel=glass&duration=3.0&seed=42",
            "sfx:water.pour?flow=strong&vessel=metal&duration=3.0&seed=42",
        ),
        (
            "sfx:water.running?flow=gentle&duration=3.0&seed=42",
            "sfx:water.running?flow=strong&duration=3.0&seed=42",
        ),
        (
            "sfx:crowd.murmur?density=sparse&duration=4.0&seed=42",
            "sfx:crowd.murmur?density=busy&duration=4.0&seed=42",
        ),
        (
            "sfx:thunder?intensity=light&distance=distant&duration=5.0&seed=42",
            "sfx:thunder?intensity=strong&distance=near&duration=5.0&seed=42",
        ),
        (
            "sfx:car.door?action=close&size=sedan&force=light&duration=1.5&seed=42",
            "sfx:car.door?action=open&size=suv&force=firm&duration=1.5&seed=42",
        ),
        (
            "sfx:car.engine?action=idle&vehicle=compact&duration=3.0&seed=42",
            "sfx:car.engine?action=rev&vehicle=truck&duration=3.0&seed=42",
        ),
        (
            "sfx:car.passby?speed=slow&vehicle=compact&duration=3.0&seed=42",
            "sfx:car.passby?speed=fast&vehicle=truck&duration=3.0&seed=42",
        ),
    )
    for first_uri, second_uri in pairs:
        first = renderer.render_uri(first_uri)
        second = renderer.render_uri(second_uri)
        assert not np.array_equal(first.samples, second.samples)

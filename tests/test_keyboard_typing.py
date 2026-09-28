from __future__ import annotations

import numpy as np
import pytest

from sfxrender import SFXRenderer
from sfxrender._keyboard import KeyEvent, _generate_typing_events, _render_keyboard_events


@pytest.mark.parametrize("sample_rate", [8_000, 16_000, 24_000, 48_000])
def test_keyboard_typing_preserves_seeded_pcm_contract(sample_rate: int) -> None:
    renderer = SFXRenderer(sample_rate=sample_rate)
    uri = "sfx:keyboard.typing?speed=steady&force=light&duration=2&seed=73"

    first = renderer.render_uri(uri)
    repeated = renderer.render_uri(uri)
    changed = renderer.render_uri(uri.replace("seed=73", "seed=74"))

    np.testing.assert_array_equal(first.samples, repeated.samples)
    assert not np.array_equal(first.samples, changed.samples)
    assert first.sample_rate == sample_rate
    assert first.samples.ndim == 1
    assert first.samples.dtype == np.float32
    assert first.samples.size > 0
    assert np.isfinite(first.samples).all()
    assert float(np.max(np.abs(first.samples))) <= 1.0
    assert float(np.sqrt(np.mean(np.square(first.samples.astype(np.float64))))) > 1e-5


def test_typing_events_are_deterministic_and_have_distinct_releases() -> None:
    events = _generate_typing_events(duration=8.0, speed="steady", force="light", seed=73)
    repeated = _generate_typing_events(duration=8.0, speed="steady", force="light", seed=73)
    changed = _generate_typing_events(duration=8.0, speed="steady", force="light", seed=74)

    assert events == repeated
    assert events != changed
    normal_events = [event for event in events if event.key_class == "normal"]
    assert len(normal_events) > 10
    assert all(event.release_time_s > event.press_time_s for event in events)
    assert (
        len([event for event in normal_events if event.release_time_s > event.press_time_s])
        >= len(normal_events) * 0.9
    )
    assert {event.key_class for event in events} >= {"normal", "space"}


def test_typing_speed_changes_event_density() -> None:
    counts = [
        len(_generate_typing_events(duration=8.0, speed=speed, force="light", seed=73))
        for speed in ("slow", "steady", "fast")
    ]

    assert counts[0] < counts[1] < counts[2]


def test_normal_key_variants_and_special_classes_render_distinctly() -> None:
    normal = tuple(
        _render_keyboard_events(
            (KeyEvent(0.0, 0.11, "normal", variant, 0.16, 0.5),),
            duration=0.16,
            sample_rate=24_000,
            seed=73,
        )
        for variant in range(5)
    )
    special = {
        key_class: _render_keyboard_events(
            (KeyEvent(0.0, 0.11, key_class, 0, 0.16, 0.5),),
            duration=0.16,
            sample_rate=24_000,
            seed=73,
        )
        for key_class in ("space", "enter", "backspace")
    }

    assert all(not np.array_equal(normal[0], variant) for variant in normal[1:])
    assert all(not np.array_equal(normal[0], sound) for sound in special.values())
    normal_centroid = _spectral_centroid(normal[0], 24_000)
    space_centroid = _spectral_centroid(special["space"], 24_000)
    assert space_centroid < normal_centroid


def test_firm_typing_changes_level_and_timbre() -> None:
    renderer = SFXRenderer(sample_rate=24_000)
    light = renderer.render_uri(
        "sfx:keyboard.typing?speed=steady&force=light&duration=3&seed=73"
    ).samples
    firm = renderer.render_uri(
        "sfx:keyboard.typing?speed=steady&force=firm&duration=3&seed=73"
    ).samples

    light_rms = float(np.sqrt(np.mean(np.square(light.astype(np.float64)))))
    firm_rms = float(np.sqrt(np.mean(np.square(firm.astype(np.float64)))))
    assert firm_rms > light_rms
    assert not np.allclose(
        light / np.max(np.abs(light)), firm / np.max(np.abs(firm)), rtol=0.01, atol=0.01
    )


def _spectral_centroid(samples: np.ndarray, sample_rate: int) -> float:
    spectrum = np.abs(np.fft.rfft(samples.astype(np.float64)))
    frequencies = np.fft.rfftfreq(samples.size, d=1.0 / sample_rate)
    return float(np.sum(spectrum * frequencies) / np.sum(spectrum))

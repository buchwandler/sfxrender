from __future__ import annotations

import numpy as np
import pytest

from sfxrender import SFXRenderer
from sfxrender._keyboard import (
    KeyEvent,
    _build_keyboard_voices,
    _generate_typing_events,
    _render_keyboard_events,
)


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


def _power_metrics(samples: np.ndarray, sample_rate: int) -> tuple[float, float, float, float]:
    signal = samples.astype(np.float64)
    windowed = signal * np.hanning(signal.size)
    power = np.square(np.abs(np.fft.rfft(windowed)))
    frequencies = np.fft.rfftfreq(signal.size, d=1.0 / sample_rate)
    total = float(power.sum())
    if total <= 0.0:
        return 0.0, 0.0, 0.0, 0.0
    centroid = float(np.dot(frequencies, power) / total)
    positive = power[power > 0.0]
    flatness = float(np.exp(np.mean(np.log(positive))) / np.mean(positive))
    dominant = int(np.argmax(power[1:]) + 1)
    return centroid, flatness, float(power[dominant] / total), float(frequencies[dominant])


def _band_fraction(samples: np.ndarray, sample_rate: int, low_hz: float, high_hz: float) -> float:
    signal = samples.astype(np.float64)
    power = np.square(np.abs(np.fft.rfft(signal * np.hanning(signal.size))))
    frequencies = np.fft.rfftfreq(signal.size, d=1.0 / sample_rate)
    selected = power[(frequencies >= low_hz) & (frequencies < high_hz)]
    return float(selected.sum() / power.sum()) if float(power.sum()) > 0.0 else 0.0


def _window_rms(samples: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(samples.astype(np.float64)))))


def _isolated_keyboard_event(
    key_class: str = "normal",
    *,
    variant: int = 0,
    release_time_s: float = 0.26,
    speed: str = "steady",
    seed: int = 73,
    velocity: float = 0.115,
    sample_rate: int = 24_000,
 ) -> np.ndarray:
    event = KeyEvent(0.05, release_time_s, key_class, variant, velocity, 0.5)
    return _render_keyboard_events(
        (event,),
        duration=release_time_s + 0.04,
        sample_rate=sample_rate,
        seed=seed,
        speed=speed,
    )


def test_normal_key_contact_is_broadband_and_dominates_its_body_tail() -> None:
    sample_rate = 24_000
    samples = _isolated_keyboard_event()
    press_start = round(0.05 * sample_rate)
    early = samples[press_start : press_start + round(0.025 * sample_rate)]
    tail = samples[press_start + round(0.080 * sample_rate) : press_start + round(0.140 * sample_rate)]
    _, flatness, dominant_bin_fraction, _ = _power_metrics(early, sample_rate)

    assert _window_rms(early) > _window_rms(tail) * 4.0
    assert _band_fraction(early, sample_rate, 1_200.0, 5_000.0) > 0.20
    assert flatness > 0.10
    assert dominant_bin_fraction < 0.12


def test_release_is_quieter_and_brighter_than_press() -> None:
    sample_rate = 24_000
    release_time_s = 0.12
    samples = _isolated_keyboard_event(release_time_s=release_time_s)
    press_start = round(0.05 * sample_rate)
    release_start = round(release_time_s * sample_rate)
    press = samples[press_start : press_start + round(0.012 * sample_rate)]
    release = samples[release_start : release_start + round(0.010 * sample_rate)]
    press_centroid = _power_metrics(press, sample_rate)[0]
    release_centroid = _power_metrics(release, sample_rate)[0]

    assert _window_rms(release) < _window_rms(press) * 0.75
    assert release_centroid > press_centroid * 1.05


def test_special_keys_are_mechanical_variants_not_tuned_notes() -> None:
    sample_rate = 24_000
    rendered = {
        key_class: _isolated_keyboard_event(key_class)
        for key_class in ("normal", "space", "enter", "backspace")
    }
    press_start = round(0.05 * sample_rate)
    windows = {
        key: samples[press_start : press_start + round(0.025 * sample_rate)]
        for key, samples in rendered.items()
    }
    metrics = {key: _power_metrics(window, sample_rate) for key, window in windows.items()}

    assert all(
        not np.array_equal(windows["normal"], windows[key])
        for key in ("space", "enter", "backspace")
    )
    assert _band_fraction(windows["space"], sample_rate, 80.0, 1_200.0) > (
        _band_fraction(windows["normal"], sample_rate, 80.0, 1_200.0) + 0.10
    )
    assert metrics["backspace"][0] > metrics["space"][0]
    assert _window_rms(windows["backspace"]) < _window_rms(windows["space"])
    assert all(metrics[key][2] < 0.15 for key in metrics)


def test_normal_variants_vary_texture_without_a_pitched_scale() -> None:
    sample_rate = 24_000
    press_start = round(0.05 * sample_rate)
    early_windows = [
        _isolated_keyboard_event(variant=variant)[
            press_start : press_start + round(0.025 * sample_rate)
        ]
        for variant in range(5)
    ]
    metrics = [_power_metrics(window, sample_rate) for window in early_windows]

    assert all(item[1] > 0.10 and item[2] < 0.12 for item in metrics)
    peak_frequencies = [item[3] for item in metrics]
    centroids = [item[0] for item in metrics]
    assert max(peak_frequencies) - min(peak_frequencies) < 500.0
    assert max(centroids) - min(centroids) < 1_500.0
    normalized = [window / max(float(np.max(np.abs(window))), 1e-12) for window in early_windows]
    assert all(not np.allclose(normalized[0], item, atol=1e-4) for item in normalized[1:])


def test_fast_typing_tightens_damping_without_shifting_body_pitch() -> None:
    event = KeyEvent(0.05, 0.26, "normal", 0, 0.115, 0.5)
    slow = _render_keyboard_events(
        (event,), duration=0.32, sample_rate=24_000, seed=73, speed="slow"
    )
    fast = _render_keyboard_events(
        (event,), duration=0.32, sample_rate=24_000, seed=73, speed="fast"
    )
    start = round(0.05 * 24_000)
    end = start + round(0.090 * 24_000)
    slow_peak = _power_metrics(slow[start:end], 24_000)[3]
    fast_peak = _power_metrics(fast[start:end], 24_000)[3]
    fast_sequence = SFXRenderer(sample_rate=24_000).render_uri(
        "sfx:keyboard.typing?speed=fast&force=firm&duration=3&seed=73"
    ).samples
    _, flatness, dominant_bin_fraction, _ = _power_metrics(fast_sequence, 24_000)

    assert slow_peak == fast_peak
    assert not np.array_equal(slow, fast)
    assert flatness > 0.05
    assert dominant_bin_fraction < 0.08


def test_body_modes_are_stable_within_a_keyboard_identity() -> None:
    profiles = _build_keyboard_voices(sample_rate=24_000, seed=73)
    repeated = _build_keyboard_voices(sample_rate=24_000, seed=73)
    changed = _build_keyboard_voices(sample_rate=24_000, seed=74)
    frequencies = [
        tuple(mode.frequency_hz for mode in profile.body_modes.modes)
        for profile in profiles[:5]
    ]
    repeated_frequencies = [
        tuple(mode.frequency_hz for mode in profile.body_modes.modes)
        for profile in repeated[:5]
    ]
    changed_frequencies = tuple(
        mode.frequency_hz for mode in changed[0].body_modes.modes
    )

    assert all(item == frequencies[0] for item in frequencies)
    assert repeated_frequencies == frequencies
    assert changed_frequencies != frequencies[0]


def test_normal_variant_selection_avoids_recent_repetition() -> None:
    events = _generate_typing_events(
        duration=30.0, speed="fast", force="firm", seed=73
    )
    variants = [event.variant for event in events if event.key_class == "normal"]

    assert len(variants) > 20
    assert all(variant not in variants[index - 3 : index] for index, variant in enumerate(variants))

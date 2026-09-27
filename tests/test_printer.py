from __future__ import annotations

import numpy as np
import pytest

from sfxrender import SFXRenderer

_PRINTER_URIS = (
    "sfx:printer.print?pages=2&speed=normal&seed=17",
    "sfx:printer.tray_open?speed=normal&paper_load=full&seed=17",
    "sfx:printer.power_switch?state=off&seed=17",
    "sfx:printer.restart?speed=normal&seed=17",
    "sfx:printer.wake?depth=deep&seed=17",
)


def _rms(samples: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(samples.astype(np.float64)))))


def _band_energy(samples: np.ndarray, sample_rate: int, low: float, high: float) -> float:
    windowed = samples * np.hanning(samples.size)
    power = np.square(np.abs(np.fft.rfft(windowed)))
    frequencies = np.fft.rfftfreq(samples.size, 1.0 / sample_rate)
    mask = (frequencies >= low) & (frequencies < high)
    return float(power[mask].sum())


def _well_separated_activity_peaks(samples: np.ndarray, sample_rate: int) -> list[int]:
    frame_size = round(0.03 * sample_rate)
    frame_count = samples.size // frame_size
    frames = samples[: frame_count * frame_size].reshape(frame_count, frame_size)
    rms = np.sqrt(np.mean(np.square(frames.astype(np.float64)), axis=1))
    smooth = np.convolve(rms, np.ones(5, dtype=np.float64) / 5.0, mode="same")
    candidates = [
        index
        for index in range(1, smooth.size - 1)
        if smooth[index] >= smooth[index - 1]
        and smooth[index] >= smooth[index + 1]
        and smooth[index] >= 0.30 * float(np.max(smooth))
    ]
    selected: list[int] = []
    minimum_gap = round(0.35 / 0.03)
    for index in sorted(candidates, key=lambda item: float(smooth[item]), reverse=True):
        if all(abs(index - previous) >= minimum_gap for previous in selected):
            selected.append(index)
    return sorted(selected)


def test_printer_effects_are_deterministic_and_seed_sensitive() -> None:
    renderer = SFXRenderer(sample_rate=16_000)
    for uri in _PRINTER_URIS:
        first = renderer.render_uri(uri).samples
        second = renderer.render_uri(uri).samples
        np.testing.assert_array_equal(first, second)

    for uri in (
        "sfx:printer.print?pages=2&seed=17",
        "sfx:printer.tray_open?paper_load=full&seed=17",
        "sfx:printer.power_switch?state=off&seed=17",
        "sfx:printer.restart?speed=normal&seed=17",
        "sfx:printer.wake?depth=deep&seed=17",
    ):
        changed = uri.replace("seed=17", "seed=18")
        assert not np.array_equal(
            renderer.render_uri(uri).samples, renderer.render_uri(changed).samples
        )


def test_print_page_count_speed_and_activity_regions() -> None:
    renderer = SFXRenderer(sample_rate=24_000)
    one = renderer.render_uri("sfx:printer.print?pages=1&speed=normal&seed=42")
    three = renderer.render_uri("sfx:printer.print?pages=3&speed=normal&seed=42")
    assert three.duration > one.duration
    assert len(_well_separated_activity_peaks(three.samples, renderer.sample_rate)) >= 3

    durations = [
        renderer.render_uri(f"sfx:printer.print?pages=1&speed={speed}&seed=42").duration
        for speed in ("slow", "normal", "fast")
    ]
    assert durations[0] > durations[1] > durations[2]


def test_tray_speed_and_paper_load_change_the_expected_properties() -> None:
    renderer = SFXRenderer(sample_rate=24_000)
    durations = [
        renderer.render_uri(f"sfx:printer.tray_open?speed={speed}&seed=42").duration
        for speed in ("slow", "normal", "fast")
    ]
    assert durations[0] > durations[1] > durations[2]

    empty = renderer.render_uri("sfx:printer.tray_open?paper_load=empty&seed=42").samples
    full = renderer.render_uri("sfx:printer.tray_open?paper_load=full&seed=42").samples
    empty_energy = _band_energy(empty, renderer.sample_rate, 900.0, 6_000.0)
    full_energy = _band_energy(full, renderer.sample_rate, 900.0, 6_000.0)
    assert full_energy > empty_energy * 1.10


def test_power_states_are_distinct_and_off_tail_decays() -> None:
    renderer = SFXRenderer(sample_rate=24_000)
    off = renderer.render_uri("sfx:printer.power_switch?state=off&seed=42").samples
    on = renderer.render_uri("sfx:printer.power_switch?state=on&seed=42").samples
    assert not np.array_equal(off, on)
    assert _rms(off[-off.size // 4 :]) < _rms(off[: off.size // 4])


def test_wake_is_shorter_and_lighter_than_restart() -> None:
    renderer = SFXRenderer(sample_rate=24_000)
    wake = renderer.render_uri("sfx:printer.wake?depth=light&seed=42")
    deep_wake = renderer.render_uri("sfx:printer.wake?depth=deep&seed=42")
    restart = renderer.render_uri("sfx:printer.restart?speed=fast&seed=42")
    assert wake.duration < deep_wake.duration < restart.duration
    assert not np.array_equal(wake.samples, restart.samples)
    assert _rms(restart.samples) > 0.0


@pytest.mark.parametrize("sample_rate", [8_000, 16_000, 24_000, 48_000])
def test_printer_effects_obey_pcm_contract_at_supported_rates(sample_rate: int) -> None:
    renderer = SFXRenderer(sample_rate=sample_rate)
    uris = (
        "sfx:printer.print?pages=2&seed=71",
        "sfx:printer.tray_open?paper_load=full&seed=72",
        "sfx:printer.power_switch?state=on&seed=73",
        "sfx:printer.restart?speed=fast&seed=74",
        "sfx:printer.wake?depth=deep&seed=75",
    )
    for uri in uris:
        sound = renderer.render_uri(uri)
        assert sound.sample_rate == sample_rate
        assert sound.samples.ndim == 1
        assert sound.samples.size > 0
        assert sound.samples.dtype == np.float32
        assert np.isfinite(sound.samples).all()
        assert float(np.max(np.abs(sound.samples))) <= 1.0
        assert _rms(sound.samples) > 1e-5

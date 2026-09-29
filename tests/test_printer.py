from __future__ import annotations

import numpy as np
import pytest

from sfxrender import SFXRenderer
from sfxrender._printer import (
    generate_printer_model,
    render_printer_tray_close,
    render_printer_tray_open,
)

_PRINTER_URIS = (
    "sfx:printer.print?pages=2&speed=normal&seed=17",
    "sfx:printer.tray_open?speed=normal&paper_load=full&seed=17",
    "sfx:printer.tray_close?speed=normal&paper_load=full&force=firm&seed=17",
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


def test_open_and_close_reuse_seeded_tray_model() -> None:
    model = generate_printer_model(seed=42, sample_rate=24_000)
    repeated_model = generate_printer_model(seed=42, sample_rate=24_000)
    opened = render_printer_tray_open(
        sample_rate=24_000, speed="normal", paper_load="full", seed=42, model=model
    )
    closed = render_printer_tray_close(
        sample_rate=24_000,
        speed="normal",
        paper_load="full",
        force="normal",
        seed=42,
        model=model,
    )
    repeated_open = render_printer_tray_open(
        sample_rate=24_000,
        speed="normal",
        paper_load="full",
        seed=42,
        model=repeated_model,
    )
    repeated_close = render_printer_tray_close(
        sample_rate=24_000,
        speed="normal",
        paper_load="full",
        force="normal",
        seed=42,
        model=repeated_model,
    )

    assert model == repeated_model
    np.testing.assert_array_equal(opened, repeated_open)
    np.testing.assert_array_equal(closed, repeated_close)
    assert not np.array_equal(opened, closed)


def test_printer_model_identity_is_stable_for_seed_and_sample_rate() -> None:
    first = generate_printer_model(seed=42, sample_rate=24_000)
    repeated = generate_printer_model(seed=42, sample_rate=24_000)
    changed_seed = generate_printer_model(seed=43, sample_rate=24_000)
    changed_rate = generate_printer_model(seed=42, sample_rate=16_000)

    assert first == repeated
    assert first.chassis_modes == repeated.chassis_modes
    assert first.tray_modes == repeated.tray_modes
    assert first.relay_contact == repeated.relay_contact
    assert first.switch_contact == repeated.switch_contact
    assert first.tray_stop_contact == repeated.tray_stop_contact
    assert first.latch_contact == repeated.latch_contact
    assert first.motor_nominal_hz == repeated.motor_nominal_hz
    assert first.fan_band_hz == repeated.fan_band_hz
    assert first != changed_seed
    assert first != changed_rate
    assert first.chassis_modes != changed_seed.chassis_modes
    assert first.motor_nominal_hz != changed_seed.motor_nominal_hz


def test_printer_effects_are_deterministic_and_seed_sensitive() -> None:
    renderer = SFXRenderer(sample_rate=16_000)
    for uri in _PRINTER_URIS:
        first = renderer.render_uri(uri).samples
        second = renderer.render_uri(uri).samples
        np.testing.assert_array_equal(first, second)

    for uri in (
        "sfx:printer.print?pages=2&seed=17",
        "sfx:printer.tray_open?paper_load=full&seed=17",
        "sfx:printer.tray_close?paper_load=full&force=firm&seed=17",
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


def test_tray_close_speed_force_and_paper_load_are_audible() -> None:
    renderer = SFXRenderer(sample_rate=24_000)
    durations = [
        renderer.render_uri(f"sfx:printer.tray_close?speed={speed}&seed=42").duration
        for speed in ("slow", "normal", "fast")
    ]
    assert durations[0] > durations[1] > durations[2]

    gentle = renderer.render_uri(
        "sfx:printer.tray_close?paper_load=empty&force=gentle&seed=42"
    ).samples
    firm = renderer.render_uri("sfx:printer.tray_close?paper_load=empty&force=firm&seed=42").samples
    terminal_start = round(0.55 * renderer.sample_rate)
    assert _rms(firm[terminal_start:]) > _rms(gentle[terminal_start:]) * 1.10

    empty = renderer.render_uri(
        "sfx:printer.tray_close?paper_load=empty&force=normal&seed=42"
    ).samples
    full = renderer.render_uri(
        "sfx:printer.tray_close?paper_load=full&force=normal&seed=42"
    ).samples
    assert (
        _band_energy(full, renderer.sample_rate, 900.0, 6_000.0)
        > _band_energy(empty, renderer.sample_rate, 900.0, 6_000.0) * 1.10
    )


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
def test_printer_confirmation_beep_is_optional_and_deterministic(sample_rate: int) -> None:
    renderer = SFXRenderer(sample_rate=sample_rate)
    cases = (
        (
            "sfx:printer.restart?speed=normal&seed=42",
            "sfx:printer.restart?speed=normal&beep=off&seed=42",
            "sfx:printer.restart?speed=normal&beep=on&seed=42",
        ),
        (
            "sfx:printer.wake?depth=deep&seed=42",
            "sfx:printer.wake?depth=deep&beep=off&seed=42",
            "sfx:printer.wake?depth=deep&beep=on&seed=42",
        ),
    )
    for original_uri, off_uri, on_uri in cases:
        original = renderer.render_uri(original_uri).samples
        explicit_off = renderer.render_uri(off_uri).samples
        first_on = renderer.render_uri(on_uri).samples
        repeated_on = renderer.render_uri(on_uri).samples

        np.testing.assert_array_equal(original, explicit_off)
        np.testing.assert_array_equal(first_on, repeated_on)
        assert first_on.shape == original.shape
        assert not np.array_equal(first_on, original)


@pytest.mark.parametrize("sample_rate", [8_000, 16_000, 24_000, 48_000])
def test_printer_effects_obey_pcm_contract_at_supported_rates(sample_rate: int) -> None:
    renderer = SFXRenderer(sample_rate=sample_rate)
    uris = (
        "sfx:printer.print?pages=2&seed=71",
        "sfx:printer.tray_open?paper_load=full&seed=72",
        "sfx:printer.tray_close?paper_load=full&force=normal&seed=72",
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

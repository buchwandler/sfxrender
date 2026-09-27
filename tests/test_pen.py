from __future__ import annotations

import numpy as np
import pytest

from sfxrender import SFXRenderer


def _rms(samples: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(samples.astype(np.float64)))))


def _frame_rms(samples: np.ndarray, sample_rate: int, frame_seconds: float = 0.02) -> np.ndarray:
    frame_size = round(sample_rate * frame_seconds)
    frame_count = samples.size // frame_size
    frames = samples[: frame_count * frame_size].reshape(frame_count, frame_size)
    return np.sqrt(np.mean(np.square(frames.astype(np.float64)), axis=1))


def _active_run_count(samples: np.ndarray, sample_rate: int) -> int:
    levels = _frame_rms(samples, sample_rate)
    active = levels > 0.25 * float(np.percentile(levels, 90))
    boundaries = np.diff(np.concatenate(([False], active, [False])).astype(np.int8))
    return int(np.count_nonzero(boundaries == 1))


def test_pen_write_is_deterministic_and_seed_sensitive() -> None:
    renderer = SFXRenderer(sample_rate=16_000)
    uri = "sfx:pen.write?duration=1.6&speed=normal&pressure=0.55&seed=29"
    first = renderer.render_uri(uri).samples
    second = renderer.render_uri(uri).samples
    changed = renderer.render_uri(uri.replace("seed=29", "seed=30")).samples
    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, changed)


def test_pen_duration_tracks_requested_time_without_level_scaling() -> None:
    renderer = SFXRenderer(sample_rate=24_000)
    short = renderer.render_uri("sfx:pen.write?duration=0.5&speed=normal&seed=47")
    long = renderer.render_uri("sfx:pen.write?duration=2.0&speed=normal&seed=47")
    assert abs(short.duration - 0.5) <= 1.0 / renderer.sample_rate
    assert abs(long.duration - 2.0) <= 1.0 / renderer.sample_rate
    assert 0.70 < _rms(long.samples) / _rms(short.samples) < 1.40


def test_pen_pressure_increases_contact_energy() -> None:
    renderer = SFXRenderer(sample_rate=24_000)
    light = renderer.render_uri(
        "sfx:pen.write?duration=1.6&speed=normal&pressure=0.1&seed=33"
    ).samples
    heavy = renderer.render_uri(
        "sfx:pen.write?duration=1.6&speed=normal&pressure=1.0&seed=33"
    ).samples
    assert _rms(heavy) > _rms(light) * 1.20


def test_pen_speed_increases_stroke_activity_density() -> None:
    renderer = SFXRenderer(sample_rate=24_000)
    slow = renderer.render_uri("sfx:pen.write?duration=2.0&speed=slow&seed=47").samples
    fast = renderer.render_uri("sfx:pen.write?duration=2.0&speed=fast&seed=47").samples
    assert _active_run_count(fast, renderer.sample_rate) > _active_run_count(
        slow, renderer.sample_rate
    )


def test_pen_has_temporal_phrasing_not_flat_static_noise() -> None:
    renderer = SFXRenderer(sample_rate=24_000)
    samples = renderer.render_uri("sfx:pen.write?duration=2.0&seed=47").samples
    levels = _frame_rms(samples, renderer.sample_rate)
    assert float(np.percentile(levels, 90)) > float(np.percentile(levels, 25)) * 1.5
    assert np.count_nonzero(levels < 0.25 * float(np.percentile(levels, 90))) > 0


@pytest.mark.parametrize("sample_rate", [8_000, 16_000, 24_000, 48_000])
def test_pen_write_obeys_pcm_contract_at_supported_rates(sample_rate: int) -> None:
    renderer = SFXRenderer(sample_rate=sample_rate)
    sound = renderer.render_uri("sfx:pen.write?duration=1.2&speed=fast&pressure=0.65&seed=81")
    assert sound.sample_rate == sample_rate
    assert sound.samples.ndim == 1
    assert sound.samples.size > 0
    assert sound.samples.dtype == np.float32
    assert np.isfinite(sound.samples).all()
    assert float(np.max(np.abs(sound.samples))) <= 1.0
    assert _rms(sound.samples) > 1e-5

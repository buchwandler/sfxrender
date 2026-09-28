from __future__ import annotations

import numpy as np
import pytest

from sfxrender._physics.rotating import render_rotating_machine


def _peak_frequency(samples: np.ndarray, sample_rate: int) -> float:
    windowed = samples * np.hanning(samples.size)
    spectrum = np.abs(np.fft.rfft(windowed))
    frequencies = np.fft.rfftfreq(samples.size, 1.0 / sample_rate)
    mask = (frequencies >= 20.0) & (frequencies < 300.0)
    return float(frequencies[mask][np.argmax(spectrum[mask])])


@pytest.mark.parametrize("sample_rate", [8_000, 16_000, 24_000, 48_000])
def test_rotating_machine_tracks_speed_curve_and_is_seeded(sample_rate: int) -> None:
    size = sample_rate * 2
    speed = np.linspace(42.0, 102.0, size, dtype=np.float32)
    first = render_rotating_machine(
        speed_curve_hz=speed,
        sample_rate=sample_rate,
        seed=42,
        event_index=3,
    )
    repeated = render_rotating_machine(
        speed_curve_hz=speed,
        sample_rate=sample_rate,
        seed=42,
        event_index=3,
    )
    changed = render_rotating_machine(
        speed_curve_hz=speed,
        sample_rate=sample_rate,
        seed=43,
        event_index=3,
    )

    np.testing.assert_array_equal(first, repeated)
    assert not np.array_equal(first, changed)
    assert first.dtype == np.float32
    assert first.shape == speed.shape
    assert np.isfinite(first).all()
    assert float(np.max(np.abs(first))) <= 1.0
    early = _peak_frequency(first[sample_rate // 8 : sample_rate * 3 // 8], sample_rate)
    late = _peak_frequency(first[sample_rate * 13 // 8 : sample_rate * 15 // 8], sample_rate)
    assert late > early + 25.0


def test_rotating_machine_rejects_invalid_speed_curves() -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        render_rotating_machine(
            speed_curve_hz=np.ones((2, 2), dtype=np.float32),
            sample_rate=24_000,
            seed=1,
        )
    with pytest.raises(ValueError, match="non-negative"):
        render_rotating_machine(
            speed_curve_hz=np.asarray([-1.0, 10.0], dtype=np.float32),
            sample_rate=24_000,
            seed=1,
        )

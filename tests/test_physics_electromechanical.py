from __future__ import annotations

import numpy as np
import pytest

from sfxrender._physics.electromechanical import render_electromechanical_hum
from sfxrender._physics.rotating import render_rotating_machine


def _rms(samples: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(samples.astype(np.float64)))))


@pytest.mark.parametrize("sample_rate", [8_000, 16_000, 24_000, 48_000])
def test_electromechanical_hum_is_deterministic_and_restrained(sample_rate: int) -> None:
    size = sample_rate
    hum = render_electromechanical_hum(
        sample_rate=sample_rate,
        size=size,
        seed=42,
        amplitude=0.02,
    )
    repeated = render_electromechanical_hum(
        sample_rate=sample_rate,
        size=size,
        seed=42,
        amplitude=0.02,
    )
    changed = render_electromechanical_hum(
        sample_rate=sample_rate,
        size=size,
        seed=43,
        amplitude=0.02,
    )
    motor = render_rotating_machine(
        speed_curve_hz=np.full(size, 90.0, dtype=np.float32),
        sample_rate=sample_rate,
        seed=42,
        amplitude=0.2,
    )

    np.testing.assert_array_equal(hum, repeated)
    assert not np.array_equal(hum, changed)
    assert hum.dtype == np.float32
    assert hum.size == size
    assert np.isfinite(hum).all()
    assert float(np.max(np.abs(hum))) <= 1.0
    assert 0.005 < _rms(hum) < 0.03
    assert _rms(hum) < _rms(motor) * 0.25


def test_electromechanical_hum_validates_envelope_shape() -> None:
    with pytest.raises(ValueError, match="envelope must match"):
        render_electromechanical_hum(
            sample_rate=24_000,
            size=100,
            seed=1,
            envelope=np.zeros(99, dtype=np.float32),
        )

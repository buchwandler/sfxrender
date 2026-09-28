from __future__ import annotations

import numpy as np
import pytest

from sfxrender._physics import render_turbulence_layer


@pytest.mark.parametrize("sample_rate", [8_000, 16_000, 24_000, 48_000])
def test_turbulence_layer_is_seeded_and_finite(sample_rate: int) -> None:
    size = round(0.8 * sample_rate)
    envelope = np.linspace(0.0, 1.0, size, dtype=np.float32)
    kwargs = {
        "size": size,
        "sample_rate": sample_rate,
        "component_id": 8,
        "low_hz": 90.0,
        "high_hz": 3_200.0,
        "amplitude": 0.04,
        "variation_depth": 0.35,
        "envelope": envelope,
    }
    first = render_turbulence_layer(**kwargs, seed=42)
    repeated = render_turbulence_layer(**kwargs, seed=42)
    changed = render_turbulence_layer(**kwargs, seed=43)

    np.testing.assert_array_equal(first, repeated)
    assert not np.array_equal(first, changed)
    assert first.dtype == np.float32
    assert np.isfinite(first).all()
    assert first.size == size
    assert abs(float(first[0])) < 1e-6
    assert float(np.max(np.abs(first))) < 0.2


def test_turbulence_layer_requires_matching_envelope_size() -> None:
    with pytest.raises(ValueError, match="same size"):
        render_turbulence_layer(
            size=100,
            sample_rate=8_000,
            seed=42,
            component_id=1,
            low_hz=100.0,
            high_hz=1_000.0,
            amplitude=0.1,
            envelope=np.ones(99, dtype=np.float32),
        )

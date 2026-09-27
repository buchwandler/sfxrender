"""Tests for reusable perceptual physical model primitives."""

from __future__ import annotations

import numpy as np
import pytest

from sfxrender._physics.models import Mode, ModeBand
from sfxrender._physics.motion import eased_motion
from sfxrender._physics.resonators import generate_modal_body, resonate

_BANDS = (
    ModeBand(70.0, 180.0, (2, 3), (0.2, 0.5), (0.08, 0.2)),
    ModeBand(180.0, 700.0, (3, 5), (0.1, 0.35), (0.03, 0.12)),
    ModeBand(700.0, 2_800.0, (2, 4), (0.04, 0.2), (0.01, 0.06)),
)


def _body(seed: int, *, sample_rate: int = 24_000):
    return generate_modal_body(
        base_bands=_BANDS,
        size_scale=1.0,
        damping_scale=0.9,
        brightness_scale=1.0,
        rng=np.random.default_rng(seed),
        sample_rate=sample_rate,
    )


def test_modal_body_generation_is_seeded_bounded_and_diverse() -> None:
    first = _body(42)
    repeated = _body(42)
    changed = _body(43)

    assert first == repeated
    assert first != changed
    assert 6 <= len(first.modes) <= 12
    frequencies = np.asarray([mode.frequency_hz for mode in first.modes])
    gains = np.asarray([mode.gain for mode in first.modes])
    assert np.all(np.isfinite(frequencies))
    assert np.all(np.isfinite(gains))
    assert np.all((frequencies > 0.0) & (frequencies < 24_000 * 0.45))
    assert np.all(gains >= 0.0)
    assert all(mode.decay_s > 0.0 for mode in first.modes)
    assert np.ptp(frequencies) > 100.0
    assert float(gains.max() / gains.sum()) < 0.5


def test_modal_generation_respects_lower_sample_rate_nyquist_margin() -> None:
    body = _body(17, sample_rate=8_000)
    assert all(mode.frequency_hz < 8_000 * 0.45 for mode in body.modes)


def test_modal_body_rejects_invalid_modes_and_ranges() -> None:
    with pytest.raises(ValueError, match="frequency_hz"):
        Mode(frequency_hz=0.0, decay_s=0.1, gain=0.2)
    with pytest.raises(ValueError, match="high_hz"):
        ModeBand(100.0, 90.0, (1, 2), (0.1, 0.3), (0.02, 0.1))


def test_resonance_is_seeded_excitation_driven_float32() -> None:
    body = _body(71)
    excitation = np.zeros(2_400, dtype=np.float32)
    excitation[12] = 1.0
    first = resonate(
        excitation,
        body=body,
        sample_rate=24_000,
        rng=np.random.default_rng(99),
    )
    repeated = resonate(
        excitation,
        body=body,
        sample_rate=24_000,
        rng=np.random.default_rng(99),
    )
    np.testing.assert_array_equal(first, repeated)
    assert first.dtype == np.float32
    assert first.shape == excitation.shape
    assert np.all(np.isfinite(first))
    assert float(np.max(np.abs(first))) > 0.0


def test_eased_motion_has_finite_smooth_derivatives_and_duration() -> None:
    slow = eased_motion(
        duration_s=2.2,
        sample_rate=1_000,
        acceleration_shape=1.4,
        deceleration_shape=1.8,
    )
    fast = eased_motion(
        duration_s=0.7,
        sample_rate=1_000,
        acceleration_shape=1.4,
        deceleration_shape=1.8,
    )

    assert slow.position.size > fast.position.size
    assert slow.position[0] == 0.0
    assert slow.position[-1] == 1.0
    assert np.all(np.diff(slow.position) >= 0.0)
    for signal in (slow.position, slow.velocity, slow.acceleration):
        assert signal.dtype == np.float32
        assert np.all(np.isfinite(signal))
    assert slow.position.shape == slow.velocity.shape == slow.acceleration.shape
    assert float(np.max(slow.velocity)) > 0.0


def test_eased_motion_rejects_nonpositive_duration() -> None:
    with pytest.raises(ValueError, match="duration_s"):
        eased_motion(
            duration_s=0.0,
            sample_rate=24_000,
            acceleration_shape=1.0,
            deceleration_shape=1.0,
        )

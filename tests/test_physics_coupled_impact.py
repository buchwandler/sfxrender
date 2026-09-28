"""Tests for one contact force driving multiple structural responses."""

from __future__ import annotations

import numpy as np
import pytest

from sfxrender._physics.contact import ImpactContact, impact_force
from sfxrender._physics.modes import Mode, ModeSet
from sfxrender._physics.physical_impacts import (
    ImpactResponse,
    render_coupled_impact,
    render_force_response,
    render_physical_impact,
)


@pytest.mark.parametrize("sample_rate", [8_000, 16_000, 24_000, 48_000])
def test_force_response_is_finite_float32_and_derives_safe_decay_tail(sample_rate: int) -> None:
    contact = ImpactContact(0.03, 2.5e7, restitution=0.4)
    trace = impact_force(contact=contact, velocity_m_s=0.8, sample_rate=sample_rate)
    modes = ModeSet((Mode(740.0, 0.22, input_gain=1.0, radiation_gain=1.0),))
    rendered = render_force_response(
        trace.force_n,
        modes,
        sample_rate=sample_rate,
        rng=np.random.default_rng(42),
        microscopic_gain=0.0,
        output_gain=1.0,
    )
    expected_tail = round(0.22 * 4.5 * sample_rate)
    assert rendered.dtype == np.float32
    assert rendered.ndim == 1
    assert rendered.size == trace.force_n.size + expected_tail
    assert np.all(np.isfinite(rendered))
    assert np.any(rendered != 0.0)


def test_force_response_preserves_silence_and_rejects_invalid_force() -> None:
    modes = ModeSet((Mode(440.0, 0.1),))
    rng = np.random.default_rng(7)
    silent = render_force_response(
        np.zeros(32, dtype=np.float32),
        modes,
        sample_rate=24_000,
        rng=rng,
    )
    assert silent.dtype == np.float32
    assert silent.shape == (32,)
    assert not np.any(silent)
    for invalid in (
        np.asarray([0.0, -1.0], dtype=np.float32),
        np.asarray([0.0, np.nan], dtype=np.float32),
        np.zeros((2, 2), dtype=np.float32),
    ):
        with pytest.raises(ValueError, match="force"):
            render_force_response(
                invalid,
                modes,
                sample_rate=24_000,
                rng=np.random.default_rng(1),
            )


def test_force_response_validates_gain_and_tail() -> None:
    modes = ModeSet((Mode(440.0, 0.1),))
    force = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
    with pytest.raises(ValueError, match="tail_s"):
        render_force_response(
            force,
            modes,
            sample_rate=24_000,
            rng=np.random.default_rng(1),
            tail_s=-0.1,
        )
    with pytest.raises(ValueError, match="output_gain"):
        render_force_response(
            force,
            modes,
            sample_rate=24_000,
            rng=np.random.default_rng(1),
            output_gain=-1.0,
        )


def test_coupled_impact_sums_responses_from_one_trace_and_wrapper_is_compatible() -> None:
    sample_rate = 24_000
    contact = ImpactContact(0.02, 2.0e7, restitution=0.4)
    trace = impact_force(contact=contact, velocity_m_s=0.72, sample_rate=sample_rate)
    modes_a = ModeSet((Mode(510.0, 0.2, input_gain=1.0),))
    modes_b = ModeSet((Mode(1_420.0, 0.12, input_gain=0.8),))
    responses = (ImpactResponse(modes_a, gain=0.8), ImpactResponse(modes_b, gain=1.3))
    rngs = (np.random.default_rng(101), np.random.default_rng(102))
    actual = render_coupled_impact(trace, responses, sample_rate=sample_rate, rngs=rngs)
    expected_parts = [
        render_force_response(
            trace.force_n,
            response.modes,
            sample_rate=sample_rate,
            rng=np.random.default_rng(seed),
            microscopic_gain=response.microscopic_gain,
            output_gain=24.0 * response.gain,
        )
        for response, seed in zip(responses, (101, 102), strict=True)
    ]
    expected = np.zeros(max(part.size for part in expected_parts), dtype=np.float32)
    for part in expected_parts:
        expected[: part.size] += part
    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == np.float32
    assert np.all(np.isfinite(actual))

    wrapped = render_physical_impact(
        trace,
        modes_a,
        sample_rate=sample_rate,
        rng=np.random.default_rng(33),
        microscopic_gain=0.0,
        output_gain=1.0,
    )
    delegated = render_force_response(
        trace.force_n,
        modes_a,
        sample_rate=sample_rate,
        rng=np.random.default_rng(33),
        microscopic_gain=0.0,
        output_gain=1.0,
    )
    np.testing.assert_array_equal(wrapped, delegated)


def test_coupled_impact_requires_one_rng_per_response() -> None:
    trace = impact_force(
        contact=ImpactContact(0.02, 2.0e7),
        velocity_m_s=0.5,
        sample_rate=24_000,
    )
    responses = (ImpactResponse(ModeSet((Mode(440.0, 0.1),))),)
    with pytest.raises(ValueError, match="one generator"):
        render_coupled_impact(trace, responses, sample_rate=24_000, rngs=())


def test_impact_response_validates_gain_values() -> None:
    modes = ModeSet((Mode(440.0, 0.1),))
    with pytest.raises(ValueError, match="gain"):
        ImpactResponse(modes, gain=-1.0)
    with pytest.raises(ValueError, match="microscopic_gain"):
        ImpactResponse(modes, microscopic_gain=-0.1)

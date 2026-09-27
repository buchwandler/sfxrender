"""Numerical and perceptual invariants for compliant impacts."""

from __future__ import annotations

import numpy as np
import pytest

from sfxrender._physics.contact import ImpactContact, impact_force


def _contact(stiffness: float, restitution: float = 0.5) -> ImpactContact:
    return ImpactContact(0.035, stiffness, exponent=1.5, restitution=restitution)


def test_hunt_crossley_trace_is_bounded_finite_and_begins_ends_at_zero() -> None:
    trace = impact_force(contact=_contact(350_000.0), velocity_m_s=1.2, sample_rate=24_000)
    assert trace.force_n.dtype == np.float32
    assert trace.force_n.ndim == 1
    assert trace.force_n.size > 2
    assert np.all(np.isfinite(trace.force_n))
    assert np.all(trace.force_n >= 0.0)
    assert trace.force_n[0] == 0.0
    assert trace.force_n[-1] == 0.0
    assert np.max(trace.indentation_m) < 0.05
    assert trace.rebound_velocity_m_s > 0.0


def test_stiffer_contact_is_shorter_and_spectrally_brighter() -> None:
    soft = impact_force(contact=_contact(90_000.0), velocity_m_s=1.0, sample_rate=24_000)
    hard = impact_force(contact=_contact(1_200_000.0), velocity_m_s=1.0, sample_rate=24_000)
    assert hard.force_n.size < soft.force_n.size
    soft_centroid = np.sum(
        np.fft.rfftfreq(soft.force_n.size, 1 / 24_000) * np.abs(np.fft.rfft(soft.force_n))
    ) / np.sum(np.abs(np.fft.rfft(soft.force_n)))
    hard_centroid = np.sum(
        np.fft.rfftfreq(hard.force_n.size, 1 / 24_000) * np.abs(np.fft.rfft(hard.force_n))
    ) / np.sum(np.abs(np.fft.rfft(hard.force_n)))
    assert hard_centroid > soft_centroid


def test_restitution_controls_rebound_energy() -> None:
    low = impact_force(
        contact=_contact(350_000.0, restitution=0.2), velocity_m_s=1.0, sample_rate=24_000
    )
    high = impact_force(
        contact=_contact(350_000.0, restitution=0.85), velocity_m_s=1.0, sample_rate=24_000
    )
    assert high.rebound_velocity_m_s > low.rebound_velocity_m_s
    assert high.rebound_velocity_m_s > 0.0


def test_contact_is_deterministic_and_zero_velocity_is_silent() -> None:
    args = {
        "contact": _contact(420_000.0),
        "velocity_m_s": 0.8,
        "sample_rate": 16_000,
    }
    first = impact_force(**args)
    repeated = impact_force(**args)
    np.testing.assert_array_equal(first.force_n, repeated.force_n)
    zero = impact_force(contact=args["contact"], velocity_m_s=0.0, sample_rate=16_000)
    assert np.all(zero.force_n == 0.0)
    assert zero.rebound_velocity_m_s == 0.0


def test_invalid_contact_inputs_are_rejected() -> None:
    with pytest.raises(ValueError, match="stiffness"):
        ImpactContact(0.035, 0.0)
    contact = _contact(100_000.0)
    with pytest.raises(ValueError, match="velocity_m_s"):
        impact_force(contact=contact, velocity_m_s=-1.0, sample_rate=24_000)
    with pytest.raises(ValueError, match="sample_rate"):
        impact_force(contact=contact, velocity_m_s=1.0, sample_rate=0)

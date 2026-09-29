"""Tests for reusable strike trains and stable physical component streams."""

from __future__ import annotations

import numpy as np
import pytest

from sfxrender._physics.contact import ImpactContact
from sfxrender._physics.modes import Mode, ModeSet
from sfxrender._physics.rng import component_rng
from sfxrender._physics.strikes import (
    StrikeEvent,
    StruckResonatorPreset,
    build_contact_force_bus,
    render_strike_train,
)

_CONTACT = ImpactContact(effective_mass_kg=0.025, stiffness=2.0e7, restitution=0.38)


def _preset(frequency_hz: float, *, decay_s: float = 0.18) -> StruckResonatorPreset:
    return StruckResonatorPreset(
        modes=ModeSet((Mode(frequency_hz, decay_s, input_gain=1.0, radiation_gain=1.0),)),
        contact=_CONTACT,
        microscopic_gain=0.0,
        output_gain=1.0,
    )


def test_component_rng_is_stable_and_component_scoped() -> None:
    first = component_rng(42, 0x50484F4E, 1)
    repeated = component_rng(42, 0x50484F4E, 1)
    other = component_rng(42, 0x50484F4E, 2)
    np.testing.assert_array_equal(first.normal(size=16), repeated.normal(size=16))
    assert not np.array_equal(
        component_rng(42, 0x50484F4E, 1).normal(size=16),
        other.normal(size=16),
    )
    assert not np.array_equal(
        component_rng(42, 0x50484F4E, 1, event_index=0).normal(size=16),
        component_rng(42, 0x50484F4E, 1, event_index=1).normal(size=16),
    )


@pytest.mark.parametrize("values", [(-1, 1, 1, 0), (1, -1, 1, 0), (1, 1, -1, 0), (1, 1, 1, -1)])
def test_component_rng_rejects_negative_identity_parts(values: tuple[int, int, int, int]) -> None:
    with pytest.raises(ValueError):
        component_rng(*values)


def test_component_rng_requires_integer_identity_parts() -> None:
    with pytest.raises(TypeError, match="namespace_id"):
        component_rng(1, 2.0, 3)


def test_strike_event_validates_schedule_and_resonator_index() -> None:
    with pytest.raises(ValueError, match="time_s"):
        StrikeEvent(-0.1, 0.5)
    with pytest.raises(ValueError, match="velocity_m_s"):
        StrikeEvent(0.0, -0.5)
    with pytest.raises(ValueError, match="resonator_index"):
        StrikeEvent(0.0, 0.5, -1)


def test_force_bus_accumulates_only_events_assigned_to_its_resonator() -> None:
    events = (StrikeEvent(0.10, 0.5, 0), StrikeEvent(0.20, 0.8, 1))
    first = build_contact_force_bus(
        events,
        resonator_index=0,
        contact=_CONTACT,
        duration_s=0.5,
        sample_rate=16_000,
    )
    second = build_contact_force_bus(
        events,
        resonator_index=1,
        contact=_CONTACT,
        duration_s=0.5,
        sample_rate=16_000,
    )
    assert first.dtype == second.dtype == np.float32
    assert first.shape == second.shape == (8_000,)
    assert np.all(first[: round(0.09 * 16_000)] == 0.0)
    assert np.all(second[: round(0.19 * 16_000)] == 0.0)
    assert np.any(first > 0.0)
    assert np.any(second > 0.0)
    assert not np.array_equal(first, second)


@pytest.mark.parametrize("sample_rate", [8_000, 16_000, 24_000, 48_000])
def test_strike_train_is_deterministic_finite_float32_and_preserves_overlapping_tails(
    sample_rate: int,
) -> None:
    events = (
        StrikeEvent(0.0, 0.55, 0),
        StrikeEvent(0.04, 0.72, 0),
        StrikeEvent(0.09, 0.45, 1),
    )
    resonators = (_preset(520.0), _preset(1_240.0, decay_s=0.28))
    first = render_strike_train(
        events=events,
        resonators=resonators,
        duration_s=0.4,
        sample_rate=sample_rate,
        rngs=(component_rng(42, 1, 1), component_rng(42, 1, 2)),
    )
    repeated = render_strike_train(
        events=events,
        resonators=resonators,
        duration_s=0.4,
        sample_rate=sample_rate,
        rngs=(component_rng(42, 1, 1), component_rng(42, 1, 2)),
    )
    np.testing.assert_array_equal(first, repeated)
    assert first.dtype == np.float32
    assert first.ndim == 1
    assert first.size > round(0.4 * sample_rate)
    assert np.all(np.isfinite(first))
    assert np.any(first[: round(0.4 * sample_rate)] != 0.0)
    assert np.any(first[round(0.4 * sample_rate) :] != 0.0)


def test_strike_train_rejects_missing_resonator_and_rng() -> None:
    with pytest.raises(ValueError, match="missing resonator"):
        render_strike_train(
            events=(StrikeEvent(0.0, 0.5, 1),),
            resonators=(_preset(440.0),),
            duration_s=0.2,
            sample_rate=24_000,
            rngs=(np.random.default_rng(1),),
        )
    with pytest.raises(ValueError, match="one generator"):
        render_strike_train(
            events=(),
            resonators=(_preset(440.0),),
            duration_s=0.2,
            sample_rate=24_000,
            rngs=(),
        )

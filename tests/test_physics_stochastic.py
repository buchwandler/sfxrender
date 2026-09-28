from __future__ import annotations

import pytest

from sfxrender._physics import sample_stochastic_events


@pytest.mark.parametrize("sample_rate", [8_000, 16_000, 24_000, 48_000])
def test_stochastic_events_are_seeded_and_bounded(sample_rate: int) -> None:
    kwargs = {
        "duration_s": 8.0,
        "sample_rate": sample_rate,
        "rate_hz": 1.5,
        "component_id": 7,
        "event_duration_s": (0.08, 0.6),
    }
    first = sample_stochastic_events(**kwargs, seed=42)
    repeated = sample_stochastic_events(**kwargs, seed=42)
    changed = sample_stochastic_events(**kwargs, seed=43)

    assert first == repeated
    assert first != changed
    assert first
    assert [event.start_sample for event in first] == sorted(event.start_sample for event in first)
    for event in first:
        assert 0 <= event.start_sample < round(kwargs["duration_s"] * sample_rate)
        assert 0 < event.duration_samples <= round(kwargs["duration_s"] * sample_rate)
        assert event.start_sample + event.duration_samples <= round(
            kwargs["duration_s"] * sample_rate
        )
        assert 0.45 <= event.level <= 1.0


def test_stochastic_event_components_have_independent_populations() -> None:
    kwargs = {
        "duration_s": 10.0,
        "sample_rate": 24_000,
        "rate_hz": 2.0,
        "seed": 42,
    }
    first = sample_stochastic_events(**kwargs, component_id=1)
    other_component = sample_stochastic_events(**kwargs, component_id=2)
    assert first != other_component


def test_zero_rate_produces_no_events() -> None:
    events = sample_stochastic_events(
        duration_s=3.0,
        sample_rate=24_000,
        rate_hz=0.0,
        seed=42,
        component_id=1,
    )
    assert events == ()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"duration_s": 0.0, "sample_rate": 24_000, "rate_hz": 1.0},
        {"duration_s": 1.0, "sample_rate": 0, "rate_hz": 1.0},
        {"duration_s": 1.0, "sample_rate": 24_000, "rate_hz": -1.0},
        {
            "duration_s": 1.0,
            "sample_rate": 24_000,
            "rate_hz": 1.0,
            "event_duration_s": (0.2, 0.1),
        },
    ],
)
def test_stochastic_event_inputs_are_validated(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        sample_stochastic_events(seed=42, component_id=1, **kwargs)

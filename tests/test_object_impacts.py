from __future__ import annotations

import numpy as np
import pytest

from sfxrender import SFXRenderer
from sfxrender._object_effects import render_object_set_down

_EFFECT_URIS = (
    "sfx:switch.toggle?state=on&seed=42",
    "sfx:button.press?size=small&force=normal&seed=42",
    "sfx:object.set_down?object=wood&surface=wood&force=normal&seed=42",
    "sfx:glass.clink?style=wine&count=1&force=normal&seed=42",
)


@pytest.mark.parametrize("sample_rate", [8_000, 16_000, 24_000, 48_000])
@pytest.mark.parametrize("uri", _EFFECT_URIS)
def test_object_effects_obey_seeded_pcm_contract(sample_rate: int, uri: str) -> None:
    renderer = SFXRenderer(sample_rate=sample_rate)
    first = renderer.render_uri(uri)
    repeated = renderer.render_uri(uri)
    changed = renderer.render_uri(uri.replace("seed=42", "seed=43"))

    np.testing.assert_array_equal(first.samples, repeated.samples)
    assert not np.array_equal(first.samples, changed.samples)
    assert first.sample_rate == sample_rate
    assert first.samples.ndim == 1
    assert first.samples.dtype == np.float32
    assert first.samples.size > 0
    assert np.isfinite(first.samples).all()
    assert float(np.max(np.abs(first.samples))) <= 1.0
    assert float(np.sqrt(np.mean(np.square(first.samples.astype(np.float64))))) > 1e-5



def test_set_down_force_scales_one_coupled_contact() -> None:
    gentle = render_object_set_down(
        sample_rate=24_000,
        object_type="wood",
        surface="wood",
        force="gentle",
        seed=42,
    )
    firm = render_object_set_down(
        sample_rate=24_000,
        object_type="wood",
        surface="wood",
        force="firm",
        seed=42,
    )
    gentle_rms = float(np.sqrt(np.mean(np.square(gentle.astype(np.float64)))))
    firm_rms = float(np.sqrt(np.mean(np.square(firm.astype(np.float64)))))
    assert firm_rms > gentle_rms * 1.5

def test_semantic_object_parameters_change_coupled_responses() -> None:
    renderer = SFXRenderer(sample_rate=24_000)
    pairs = (
        (
            "sfx:switch.toggle?state=on&seed=42",
            "sfx:switch.toggle?state=off&seed=42",
        ),
        (
            "sfx:button.press?size=small&force=normal&seed=42",
            "sfx:button.press?size=large&force=firm&seed=42",
        ),
        (
            "sfx:object.set_down?object=wood&surface=wood&force=gentle&seed=42",
            "sfx:object.set_down?object=ceramic&surface=stone&force=firm&seed=42",
        ),
        (
            "sfx:glass.clink?style=wine&count=1&force=gentle&seed=42",
            "sfx:glass.clink?style=tumbler&count=2&force=firm&seed=42",
        ),
    )
    for first_uri, second_uri in pairs:
        first = renderer.render_uri(first_uri).samples
        second = renderer.render_uri(second_uri).samples
        assert not np.array_equal(first, second)


def test_glass_clink_count_adds_a_separate_contact() -> None:
    renderer = SFXRenderer(sample_rate=24_000)
    one = renderer.render_uri("sfx:glass.clink?count=1&seed=42")
    two = renderer.render_uri("sfx:glass.clink?count=2&seed=42")
    assert two.duration > one.duration + 0.15

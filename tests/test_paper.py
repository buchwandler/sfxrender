from __future__ import annotations

import numpy as np
import pytest

from sfxrender import SFXRenderer

_PAPER_URIS = (
    "sfx:paper.page_turn?pages=1&speed=normal&seed=42",
    "sfx:paper.handle?duration=0.8&intensity=normal&seed=42",
)


@pytest.mark.parametrize("sample_rate", [8_000, 16_000, 24_000, 48_000])
@pytest.mark.parametrize("uri", _PAPER_URIS)
def test_paper_effects_obey_seeded_pcm_contract(sample_rate: int, uri: str) -> None:
    renderer = SFXRenderer(sample_rate=sample_rate)
    first = renderer.render_uri(uri)
    repeated = renderer.render_uri(uri)
    changed = renderer.render_uri(uri.replace("seed=42", "seed=43"))

    np.testing.assert_array_equal(first.samples, repeated.samples)
    assert not np.array_equal(first.samples, changed.samples)
    assert first.sample_rate == sample_rate
    assert first.samples.dtype == np.float32
    assert first.samples.ndim == 1
    assert first.samples.size > 0
    assert np.isfinite(first.samples).all()
    assert float(np.max(np.abs(first.samples))) <= 1.0
    assert float(np.max(np.abs(first.samples))) > 1e-5


def test_page_count_and_speed_change_turn_timing() -> None:
    renderer = SFXRenderer(sample_rate=24_000)
    one_slow = renderer.render_uri("sfx:paper.page_turn?pages=1&speed=slow&seed=42")
    four_fast = renderer.render_uri("sfx:paper.page_turn?pages=4&speed=fast&seed=42")
    one_fast = renderer.render_uri("sfx:paper.page_turn?pages=1&speed=fast&seed=42")
    assert four_fast.duration > one_fast.duration + 0.5
    assert one_slow.duration > one_fast.duration


def test_paper_handling_duration_and_intensity_are_semantic() -> None:
    renderer = SFXRenderer(sample_rate=24_000)
    gentle = renderer.render_uri(
        "sfx:paper.handle?duration=0.8&intensity=gentle&seed=42"
    )
    rough = renderer.render_uri("sfx:paper.handle?duration=0.8&intensity=rough&seed=42")
    longer = renderer.render_uri("sfx:paper.handle?duration=1.6&intensity=gentle&seed=42")
    assert not np.array_equal(gentle.samples, rough.samples)
    assert longer.duration > gentle.duration + 0.7

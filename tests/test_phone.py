"""Tests for stable mechanical and electronic phone/chime sources."""

from __future__ import annotations

import numpy as np
import pytest

from sfxrender import SFXRenderer
from sfxrender._doorbell import generate_door_chime, render_door_chime
from sfxrender._electronic import TonePreset, TonePulse, render_tone_pattern
from sfxrender._phone import (
    generate_classic_phone_ringer,
    render_classic_phone_ring,
    render_electronic_phone_ring,
)
from sfxrender._physics.rng import component_rng


def test_classic_phone_model_identity_is_seeded_and_stable() -> None:
    first = generate_classic_phone_ringer(seed=42, sample_rate=24_000)
    repeated = generate_classic_phone_ringer(seed=42, sample_rate=24_000)
    changed = generate_classic_phone_ringer(seed=43, sample_rate=24_000)
    assert first == repeated
    assert first != changed
    assert first.gong_a.modes != first.gong_b.modes


def test_classic_phone_ring_has_repeated_transient_strikes_and_seed_variation() -> None:
    first_model = generate_classic_phone_ringer(seed=42, sample_rate=16_000)
    second_model = generate_classic_phone_ringer(seed=43, sample_rate=16_000)
    first = render_classic_phone_ring(first_model, duration_s=0.62, sample_rate=16_000)
    repeated = render_classic_phone_ring(first_model, duration_s=0.62, sample_rate=16_000)
    changed = render_classic_phone_ring(second_model, duration_s=0.62, sample_rate=16_000)
    np.testing.assert_array_equal(first, repeated)
    assert not np.array_equal(first, changed)
    assert first.dtype == np.float32
    assert np.all(np.isfinite(first))
    assert np.count_nonzero(np.abs(first) > 1e-5) > 16_000


def test_electronic_phone_ring_uses_repeatable_tone_pulses() -> None:
    first = render_electronic_phone_ring(seed=42, duration_s=0.42, sample_rate=24_000)
    repeated = render_electronic_phone_ring(seed=42, duration_s=0.42, sample_rate=24_000)
    changed = render_electronic_phone_ring(seed=43, duration_s=0.42, sample_rate=24_000)
    np.testing.assert_array_equal(first, repeated)
    assert not np.array_equal(first, changed)
    assert first.dtype == np.float32
    assert np.all(np.isfinite(first))


def test_mechanical_doorbell_is_seeded_and_two_bar_chime_is_distinct() -> None:
    model = generate_door_chime(seed=71, sample_rate=24_000)
    repeated_model = generate_door_chime(seed=71, sample_rate=24_000)
    assert model == repeated_model
    first = render_door_chime(model, duration_s=0.8, sample_rate=24_000)
    repeated = render_door_chime(repeated_model, duration_s=0.8, sample_rate=24_000)
    np.testing.assert_array_equal(first, repeated)
    assert first.dtype == np.float32
    assert np.all(np.isfinite(first))
    assert np.any(first != 0.0)


def test_tone_source_validates_presets_and_renders_supported_sample_rates() -> None:
    preset = TonePreset(740.0, (0.8, 0.2), 0.01, 0.04, bandwidth_hz=(200.0, 2_800.0))
    for sample_rate in (8_000, 16_000, 24_000, 48_000):
        sound = render_tone_pattern(
            preset=preset,
            pulses=(TonePulse(0.0, 0.2),),
            duration_s=0.25,
            sample_rate=sample_rate,
            rng=component_rng(42, 7, 1),
        )
        assert sound.dtype == np.float32
        assert sound.size == round(0.25 * sample_rate)
        assert np.all(np.isfinite(sound))
        assert np.any(sound != 0.0)
    with pytest.raises(ValueError, match="frequency_hz"):
        TonePreset(0.0, (1.0,), 0.0, 0.0)
    with pytest.raises(ValueError, match="duration_s"):
        TonePulse(0.0, 0.0)


def test_phone_and_doorbell_keep_public_phone_uri_and_catalog_contracts() -> None:
    renderer = SFXRenderer(sample_rate=16_000)
    classic = renderer.render_uri("sfx:phone.ring?style=classic&count=2&interval=0.8&seed=42")
    electronic = renderer.render_uri("sfx:phone.ring?style=electronic&count=2&interval=0.8&seed=42")
    assert classic.samples.size > electronic.samples.size
    assert not np.array_equal(classic.samples, electronic.samples)
    assert renderer.validate_uri("sfx:doorbell.ring?style=chime&count=2&interval=0.8&seed=42")
    assert renderer.validate_uri("sfx:doorbell.ring?style=electronic&count=1&seed=43")
    chime = renderer.render_uri("sfx:doorbell.ring?style=chime&count=1&seed=42")
    electronic_chime = renderer.render_uri("sfx:doorbell.ring?style=electronic&count=1&seed=42")
    assert not np.array_equal(chime.samples, electronic_chime.samples)

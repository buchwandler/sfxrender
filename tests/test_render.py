from pathlib import Path

import numpy as np
import pytest

from sfxrender import SFXRenderer


def test_knock_is_deterministic() -> None:
    renderer = SFXRenderer(sample_rate=16_000)
    uri = "sfx:impact.knock?material=oak&count=3&force=0.7&seed=42"
    first = renderer.render_uri(uri)
    second = renderer.render_uri(uri)
    np.testing.assert_array_equal(first.samples, second.samples)
    assert first.samples.dtype == np.float32
    assert first.sample_rate == 16_000
    assert first.duration > 0.4


def test_different_seed_changes_output() -> None:
    renderer = SFXRenderer()
    first = renderer.render_uri("sfx:footsteps.walk?count=3&seed=1")
    second = renderer.render_uri("sfx:footsteps.walk?count=3&seed=2")
    assert not np.array_equal(first.samples, second.samples)


def test_builtin_effects_render() -> None:
    renderer = SFXRenderer(sample_rate=12_000)
    uris = [
        "sfx:impact.knock?seed=1",
        "sfx:footsteps.walk?surface=gravel&count=4&seed=2",
        "sfx:phone.ring?style=classic&count=2&seed=3",
        "sfx:door.open?material=wood&speed=slow&seed=4",
    ]
    for uri in uris:
        sound = renderer.render_uri(uri)
        assert sound.samples.ndim == 1
        assert sound.samples.size > 0
        assert float(np.max(np.abs(sound.samples))) <= 1.0


def test_write_wav(tmp_path: Path) -> None:
    renderer = SFXRenderer(sample_rate=8_000)
    sound = renderer.render_uri("sfx:impact.knock?seed=42")
    output = sound.write_wav(tmp_path / "knock.wav")
    assert output.is_file()
    assert output.stat().st_size > 44


def _rms(samples: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))


@pytest.mark.parametrize(
    ("effect", "parameters"),
    [
        ("impact.knock", "material=oak&count=3"),
        ("footsteps.walk", "surface=wood&count=4"),
    ],
)
def test_force_has_meaningful_level_effect(effect: str, parameters: str) -> None:
    renderer = SFXRenderer()
    low = renderer.render_uri(f"sfx:{effect}?{parameters}&force=0.2&seed=42")
    high = renderer.render_uri(f"sfx:{effect}?{parameters}&force=0.9&seed=42")

    assert _rms(high.samples) > _rms(low.samples) * 2.0
    assert np.max(np.abs(high.samples)) > np.max(np.abs(low.samples))


def test_phone_seed_changes_audible_rendering() -> None:
    renderer = SFXRenderer()
    first = renderer.render_uri("sfx:phone.ring?style=classic&count=2&seed=21")
    second = renderer.render_uri("sfx:phone.ring?style=classic&count=2&seed=22")

    assert _rms(first.samples - second.samples) > 0.01


def test_all_builtins_are_finite_float32_mono_and_wav_bounded() -> None:
    renderer = SFXRenderer()
    for effect in renderer.effects():
        for uri in (f"sfx:{effect}", f"sfx:{effect}?seed=123"):
            first = renderer.render_uri(uri)
            again = renderer.render_uri(uri)
            np.testing.assert_array_equal(first.samples, again.samples)
            assert first.samples.ndim == 1
            assert first.samples.dtype == np.float32
            assert np.all(np.isfinite(first.samples))
            assert float(np.max(np.abs(first.samples))) <= 1.0


@pytest.mark.parametrize(
    ("uri", "message"),
    [
        ("sfx:impact.knock?count=three", "count must be an integer"),
        ("sfx:impact.knock?force=loud", "force must be a number"),
    ],
)
def test_malformed_numeric_parameters_have_context(uri: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        SFXRenderer().render_uri(uri)

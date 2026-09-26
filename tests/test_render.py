from pathlib import Path

import numpy as np

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

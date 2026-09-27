from pathlib import Path

import numpy as np
import pytest

from sfxrender import SFXRenderer
from sfxrender.procedural import _event_starts
from sfxrender.types import SfxSpec


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


def _band_energy(samples: np.ndarray, sample_rate: int, low: float, high: float) -> float:
    windowed = samples * np.hanning(samples.size)
    power = np.square(np.abs(np.fft.rfft(windowed)))
    frequencies = np.fft.rfftfreq(samples.size, 1.0 / sample_rate)
    mask = (frequencies >= low) & (frequencies < high)
    return float(power[mask].sum())


def _band_fraction(samples: np.ndarray, sample_rate: int, low: float, high: float) -> float:
    windowed = samples * np.hanning(samples.size)
    power = np.square(np.abs(np.fft.rfft(windowed)))
    frequencies = np.fft.rfftfreq(samples.size, 1.0 / sample_rate)
    mask = (frequencies >= low) & (frequencies < high)
    return float(power[mask].sum() / power.sum())


def test_event_starts_are_seeded_humanized_and_bounded() -> None:
    spec = SfxSpec("footsteps.walk", {"seed": "42"})
    other_spec = SfxSpec("footsteps.walk", {"seed": "43"})
    starts = _event_starts(
        count=9, interval=0.52, sample_rate=16_000, spec=spec, jitter_fraction=0.04
    )
    assert starts == _event_starts(
        count=9, interval=0.52, sample_rate=16_000, spec=spec, jitter_fraction=0.04
    )
    assert starts != _event_starts(
        count=9, interval=0.52, sample_rate=16_000, spec=other_spec, jitter_fraction=0.04
    )
    gaps = np.diff(starts) / 16_000
    assert np.all(gaps >= 0.86 * 0.52 - 1 / 16_000)
    assert np.all(gaps <= 1.14 * 0.52 + 1 / 16_000)
    assert not np.allclose(gaps, gaps[0], rtol=0.0, atol=1 / 16_000)


def test_repeated_footsteps_have_distinct_event_shapes() -> None:
    sample_rate = 16_000
    spec = SfxSpec("footsteps.walk", {"seed": "42"})
    starts = _event_starts(
        count=4, interval=0.6, sample_rate=sample_rate, spec=spec, jitter_fraction=0.04
    )
    rendered = SFXRenderer(sample_rate=sample_rate).render_uri(
        "sfx:footsteps.walk?surface=wood&footwear=boots&count=4&interval=0.6&seed=42"
    )
    event_length = int(0.24 * sample_rate)
    events = np.stack([rendered.samples[start : start + event_length] for start in starts])
    correlations = np.corrcoef(events)
    off_diagonal = correlations[np.triu_indices(events.shape[0], k=1)]
    assert float(np.max(np.abs(off_diagonal))) < 0.98


def test_footsteps_keep_delayed_sole_energy_after_initial_contact() -> None:
    sample_rate = 16_000
    sound = SFXRenderer(sample_rate=sample_rate).render_uri(
        "sfx:footsteps.walk?surface=wood&footwear=boots&count=1&seed=42"
    )
    early = sound.samples[: int(0.035 * sample_rate)]
    delayed = sound.samples[int(0.035 * sample_rate) : int(0.12 * sample_rate)]
    assert _rms(delayed) > _rms(early) * 0.25


def test_footstep_surface_and_footwear_have_relative_spectral_signatures() -> None:
    sample_rate = 16_000
    renderer = SFXRenderer(sample_rate=sample_rate)
    wood = renderer.render_uri(
        "sfx:footsteps.walk?surface=wood&footwear=shoes&count=1&seed=42"
    ).samples
    stone = renderer.render_uri(
        "sfx:footsteps.walk?surface=stone&footwear=shoes&count=1&seed=42"
    ).samples
    carpet_barefoot = renderer.render_uri(
        "sfx:footsteps.walk?surface=carpet&footwear=barefoot&count=1&seed=42"
    ).samples
    stone_heels = renderer.render_uri(
        "sfx:footsteps.walk?surface=stone&footwear=heels&count=1&seed=42"
    ).samples
    wood_body = _band_fraction(wood, sample_rate, 200.0, 1000.0)
    stone_body = _band_fraction(stone, sample_rate, 200.0, 1000.0)
    carpet_high = _band_fraction(carpet_barefoot, sample_rate, 2000.0, 8000.0)
    heel_high = _band_fraction(stone_heels, sample_rate, 2000.0, 8000.0)
    assert wood_body > stone_body * 1.5
    assert carpet_high < heel_high * 0.65


def test_metal_knock_has_more_high_frequency_decay_than_wall() -> None:
    sample_rate = 16_000
    renderer = SFXRenderer(sample_rate=sample_rate)
    wall = renderer.render_uri("sfx:impact.knock?material=wall&count=1&seed=42").samples
    metal = renderer.render_uri("sfx:impact.knock?material=metal&count=1&seed=42").samples
    start = int(0.03 * sample_rate)
    end = int(0.20 * sample_rate)
    wall_tail = _band_energy(wall[start:end], sample_rate, 1000.0, 8000.0)
    metal_tail = _band_energy(metal[start:end], sample_rate, 1000.0, 8000.0)
    assert metal_tail > wall_tail * 1.5


def test_layered_effect_durations_stay_within_gait_and_hit_bounds() -> None:
    renderer = SFXRenderer(sample_rate=16_000)
    footsteps = renderer.render_uri(
        "sfx:footsteps.walk?surface=wood&footwear=boots&count=8&interval=0.52&seed=42"
    )
    knocks = renderer.render_uri("sfx:impact.knock?material=oak&count=3&interval=0.25&seed=42")
    assert footsteps.duration <= 8 * 0.52 + 0.4
    assert footsteps.duration >= 7 * 0.52 * 0.86 + 0.24
    assert knocks.duration <= 3 * 0.25 + 0.8


def test_repeated_knocks_have_small_event_variations() -> None:
    sample_rate = 16_000
    spec = SfxSpec("impact.knock", {"seed": "42"})
    starts = _event_starts(
        count=4, interval=0.65, sample_rate=sample_rate, spec=spec, jitter_fraction=0.008
    )
    rendered = SFXRenderer(sample_rate=sample_rate).render_uri(
        "sfx:impact.knock?material=wood&count=4&interval=0.65&seed=42"
    )
    event_length = int(0.20 * sample_rate)
    events = np.stack([rendered.samples[start : start + event_length] for start in starts])
    correlations = np.corrcoef(events)
    off_diagonal = correlations[np.triu_indices(events.shape[0], k=1)]
    assert float(np.max(np.abs(off_diagonal))) < 0.99


def test_knock_materials_have_distinct_decay_lengths() -> None:
    renderer = SFXRenderer(sample_rate=16_000)
    wood = renderer.render_uri("sfx:impact.knock?material=wood&count=1&seed=42")
    oak = renderer.render_uri("sfx:impact.knock?material=oak&count=1&seed=42")
    wall = renderer.render_uri("sfx:impact.knock?material=wall&count=1&seed=42")
    assert oak.duration > wood.duration
    assert wall.duration < wood.duration

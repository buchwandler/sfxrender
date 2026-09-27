from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import pytest

from sfxrender import (
    InvalidEffectParameterError,
    SFXRenderer,
    UnknownEffectError,
    __version__,
)
from sfxrender._foley_profiles import AGGREGATE_SURFACE_PROFILES
from sfxrender._footsteps import _sample_particle_events, _synthetic_grf
from sfxrender.procedural import _event_starts
from sfxrender.types import SfxSpec


def test_knock_is_deterministic() -> None:
    renderer = SFXRenderer(sample_rate=16_000)
    uri = "sfx:impact.knock?material=oak&count=3&force=0.7&seed=42"
    first = renderer.render_uri(uri)
    second = renderer.render_uri(uri)
    assert first.samples.tobytes() == second.samples.tobytes()


def test_seed_changes_stochastic_output() -> None:
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
        "sfx:door.close?material=wood&speed=normal&force=0.7&seed=5",
    ]
    for uri in uris:
        sound = renderer.render_uri(uri)
        assert sound.samples.ndim == 1
        assert sound.samples.size > 0
        assert float(np.max(np.abs(sound.samples))) <= 1.0


@pytest.mark.parametrize("sample_rate", [22_050, 24_000, 44_100, 48_000])
def test_requested_sample_rate_is_honored(sample_rate: int) -> None:
    rendered = SFXRenderer(sample_rate=sample_rate).render_uri(
        "sfx:impact.knock?material=oak&seed=42"
    )
    assert rendered.sample_rate == sample_rate


def test_pcm_contract_and_result_metadata() -> None:
    rendered = SFXRenderer(sample_rate=24_000).render_uri("sfx:impact.knock?seed=42")
    assert rendered.samples.dtype == np.float32
    assert rendered.samples.ndim == 1
    assert rendered.samples.size > 0
    assert np.isfinite(rendered.samples).all()
    assert float(np.max(np.abs(rendered.samples))) <= 1.0
    assert rendered.duration == rendered.samples.size / rendered.sample_rate
    assert rendered.spec.effect == "impact.knock"


def test_package_version_matches_distribution_metadata() -> None:
    try:
        installed_version = version("sfxrender")
    except PackageNotFoundError:
        assert __version__ == "0+unknown"
    else:
        assert __version__ == installed_version


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
        ("footsteps.walk", "surface=gravel&footwear=boots&count=1"),
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
    with pytest.raises(InvalidEffectParameterError, match=message):
        SFXRenderer().render_uri(uri)


def test_unknown_effect_is_a_typed_error() -> None:
    with pytest.raises(UnknownEffectError, match="unknown SFX effect 'not.real'"):
        SFXRenderer().render_uri("sfx:not.real?seed=42")


@pytest.mark.parametrize(
    ("uri", "message"),
    [
        ("sfx:impact.knock?unknown=1", "unknown parameter"),
        ("sfx:impact.knock?count=three", "count must be an integer"),
        ("sfx:impact.knock?force=4.0", "expected a value in 0.05..1.0"),
        ("sfx:impact.knock?seed=not-a-number", "non-negative integer"),
        ("sfx:impact.knock?seed=-1", "non-negative integer"),
    ],
)
def test_invalid_effect_parameters_are_typed(uri: str, message: str) -> None:
    with pytest.raises(InvalidEffectParameterError, match=message):
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


def _dominant_bin_fraction(samples: np.ndarray) -> float:
    windowed = samples * np.hanning(samples.size)
    power = np.square(np.abs(np.fft.rfft(windowed)))
    return float(power.max() / power.sum())


def _spectral_centroid(samples: np.ndarray, sample_rate: int) -> float:
    windowed = samples * np.hanning(samples.size)
    power = np.square(np.abs(np.fft.rfft(windowed)))
    frequencies = np.fft.rfftfreq(samples.size, 1.0 / sample_rate)
    return float(np.sum(frequencies * power) / np.sum(power))


def _window_rms(samples: np.ndarray, sample_rate: int, start: float, end: float) -> float:
    segment = samples[int(start * sample_rate) : int(end * sample_rate)]
    return _rms(segment)


def test_synthetic_grf_is_seeded_and_has_heel_sole_toe_phases() -> None:
    sample_rate = 16_000
    grf = _synthetic_grf(
        sample_rate=sample_rate,
        footwear="boots",
        force=0.58,
        rng=np.random.default_rng(11),
    )
    repeated = _synthetic_grf(
        sample_rate=sample_rate,
        footwear="boots",
        force=0.58,
        rng=np.random.default_rng(11),
    )
    changed = _synthetic_grf(
        sample_rate=sample_rate,
        footwear="boots",
        force=0.58,
        rng=np.random.default_rng(12),
    )
    np.testing.assert_array_equal(grf, repeated)
    assert not np.array_equal(grf, changed)
    assert 0.22 <= grf.size / sample_rate <= 0.31
    assert float(np.max(grf[: int(0.035 * sample_rate)])) > 0.0
    assert float(np.max(grf[int(0.035 * sample_rate) : int(0.12 * sample_rate)])) > 0.0
    assert float(np.max(grf[int(0.12 * sample_rate) : int(0.22 * sample_rate)])) > 0.0


def test_wood_boots_avoid_bass_note_dominance_and_keep_contact_stages() -> None:
    sample_rate = 16_000
    sound = SFXRenderer(sample_rate=sample_rate).render_uri(
        "sfx:footsteps.walk?surface=wood&footwear=boots&count=1&force=0.58&seed=11"
    )
    samples = sound.samples[: int(0.22 * sample_rate)]
    assert _band_fraction(samples, sample_rate, 80.0, 200.0) < 0.70
    assert _dominant_bin_fraction(samples) < 0.20
    assert _band_fraction(samples, sample_rate, 200.0, 1_500.0) > 0.20
    heel = _window_rms(samples, sample_rate, 0.0, 0.035)
    sole = _window_rms(samples, sample_rate, 0.035, 0.12)
    release = _window_rms(samples, sample_rate, 0.12, 0.22)
    assert heel > 0.01
    assert sole > heel * 0.5
    assert release > heel * 0.25


def test_footwear_changes_contact_weight_timing_and_brightness() -> None:
    sample_rate = 16_000
    renderer = SFXRenderer(sample_rate=sample_rate)
    sounds = {
        footwear: renderer.render_uri(
            f"sfx:footsteps.walk?surface=wood&footwear={footwear}&count=1&force=0.62&seed=17"
        ).samples
        for footwear in ("barefoot", "shoes", "boots", "heels")
    }
    assert len({samples.size for samples in sounds.values()}) == 4
    assert all(
        not np.array_equal(sounds[left], sounds[right])
        for left, right in (("barefoot", "shoes"), ("shoes", "boots"), ("boots", "heels"))
    )
    assert _rms(sounds["boots"]) > _rms(sounds["heels"]) * 1.5
    assert _band_fraction(sounds["heels"], sample_rate, 2_000.0, 8_000.0) > (
        _band_fraction(sounds["boots"], sample_rate, 2_000.0, 8_000.0) * 1.05
    )


def test_gravel_micro_impact_density_tracks_grf_and_energy_is_heavy_tailed() -> None:
    sample_rate = 16_000
    grf = np.concatenate(
        (np.full(800, 0.85, dtype=np.float32), np.full(800, 0.08, dtype=np.float32))
    )
    events = _sample_particle_events(
        grf=grf,
        surface=AGGREGATE_SURFACE_PROFILES["gravel"],
        rng=np.random.default_rng(91),
        sample_rate=sample_rate,
    )
    high_load = sum(event.start_sample < 800 for event in events)
    low_load = sum(event.start_sample >= 800 for event in events)
    energies = np.asarray([event.energy for event in events])
    profile = AGGREGATE_SURFACE_PROFILES["gravel"]
    assert high_load > low_load
    assert np.all(energies >= profile.minimum_energy)
    assert np.all(energies <= profile.maximum_energy)
    assert float(np.median(energies)) < (profile.minimum_energy + profile.maximum_energy) / 2


def test_gravel_is_seeded_stochastic_and_broader_than_carpet() -> None:
    sample_rate = 16_000
    renderer = SFXRenderer(sample_rate=sample_rate)
    gravel_uri = "sfx:footsteps.walk?surface=gravel&footwear=shoes&count=1&seed=18"
    gravel = renderer.render_uri(gravel_uri).samples
    repeated = renderer.render_uri(gravel_uri).samples
    changed = renderer.render_uri(
        "sfx:footsteps.walk?surface=gravel&footwear=shoes&count=1&seed=19"
    ).samples
    carpet = renderer.render_uri(
        "sfx:footsteps.walk?surface=carpet&footwear=shoes&count=1&seed=18"
    ).samples
    np.testing.assert_array_equal(gravel, repeated)
    assert not np.array_equal(gravel, changed)
    assert _spectral_centroid(gravel, sample_rate) > (
        _spectral_centroid(carpet, sample_rate) * 1.25
    )


def test_knocks_do_not_collapse_to_a_single_hollow_mode() -> None:
    renderer = SFXRenderer(sample_rate=16_000)
    for material in ("wood", "oak", "wall", "metal"):
        sound = renderer.render_uri(f"sfx:impact.knock?material={material}&count=1&seed=31")
        assert _dominant_bin_fraction(sound.samples) < 0.20

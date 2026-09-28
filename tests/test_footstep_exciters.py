from __future__ import annotations


from dataclasses import replace
import numpy as np

from sfxrender import SFXRenderer
from sfxrender._footsteps import (
    FrictionEvent,
    _solid_footstep,
    generate_footstep_exciter,
)
from sfxrender._foley_profiles import FOOTWEAR_PROFILES
from sfxrender._physics.presets import FOOTWEAR_CONTACTS


def test_single_step_changes_with_interval() -> None:
    renderer = SFXRenderer(sample_rate=24_000)
    slow = renderer.render_uri(
        "sfx:footsteps.walk?surface=wood&footwear=shoes&count=1&force=0.6&interval=0.80&seed=73"
    )
    fast = renderer.render_uri(
        "sfx:footsteps.walk?surface=wood&footwear=shoes&count=1&force=0.6&interval=0.35&seed=73"
    )

    assert not np.array_equal(slow.samples, fast.samples)


def test_exciter_markers_follow_envelope_and_pace_changes_contact_shape() -> None:
    def build(interval: float):
        return generate_footstep_exciter(
            sample_rate=24_000,
            footwear="boots",
            force=0.6,
            gait="walk",
            step_interval_s=interval,
            rng=np.random.default_rng(73),
        )

    slow = build(0.80)
    fast = build(0.35)

    assert fast.envelope.size < slow.envelope.size
    assert fast.load_scale > slow.load_scale
    assert {impact.role for impact in slow.impacts} >= {"heel", "sole"}
    assert all(slow.envelope[round(impact.time_s * 24_000)] > 0.0 for impact in slow.impacts)


def test_footwear_hardness_changes_early_transient_energy() -> None:
    renderer = SFXRenderer(sample_rate=24_000)
    heels = renderer.render_uri(
        "sfx:footsteps.walk?surface=wood&footwear=heels&count=1&force=0.6&seed=73"
    ).samples
    barefoot = renderer.render_uri(
        "sfx:footsteps.walk?surface=wood&footwear=barefoot&count=1&force=0.6&seed=73"
    ).samples

    assert _early_high_frequency_energy(heels, 24_000) > _early_high_frequency_energy(
        barefoot, 24_000
    )

    assert _early_high_frequency_fraction(heels, 24_000) > 0.01


def test_forced_scuff_adds_friction_without_changing_earlier_impacts() -> None:
    sample_rate = 24_000
    exciter = generate_footstep_exciter(
        sample_rate=sample_rate,
        footwear="boots",
        force=0.6,
        gait="walk",
        step_interval_s=0.52,
        rng=np.random.default_rng(73),
    )
    quiet_exciter = replace(exciter, friction_events=())
    scuffed_exciter = replace(
        exciter,
        friction_events=(FrictionEvent(0.16, 0.045, 0.7, 1.1, "scuff"),),
    )

    def render(event_exciter) -> np.ndarray:
        return _solid_footstep(
            exciter=event_exciter,
            surface="wood",
            footwear=FOOTWEAR_PROFILES["boots"],
            footwear_contact=FOOTWEAR_CONTACTS["boots"],
            rng=np.random.default_rng(82),
            sample_rate=sample_rate,
        )

    quiet = render(quiet_exciter)
    scuffed = render(scuffed_exciter)
    prefix = round(0.15 * sample_rate)
    np.testing.assert_array_equal(quiet[:prefix], scuffed[:prefix])
    assert not np.array_equal(quiet, scuffed)


def _early_high_frequency_energy(samples: np.ndarray, sample_rate: int) -> float:
    segment = samples[: round(0.075 * sample_rate)].astype(np.float64)
    spectrum = np.abs(np.fft.rfft(segment * np.hanning(segment.size)))
    frequencies = np.fft.rfftfreq(segment.size, d=1.0 / sample_rate)
    return float(np.sum(np.square(spectrum[frequencies >= 2_400.0])))


def _early_high_frequency_fraction(samples: np.ndarray, sample_rate: int) -> float:
    segment = samples[: round(0.075 * sample_rate)].astype(np.float64)
    spectrum = np.abs(np.fft.rfft(segment * np.hanning(segment.size)))
    frequencies = np.fft.rfftfreq(segment.size, d=1.0 / sample_rate)
    power = np.square(spectrum)
    return float(np.sum(power[frequencies >= 2_400.0]) / np.sum(power))

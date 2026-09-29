"""Mechanical chime and electronic doorbell sources."""

from __future__ import annotations

from dataclasses import dataclass

from ._electronic import TonePreset, TonePulse, render_tone_pattern
from ._physics.contact import ImpactContact
from ._physics.modes import Mode, ModeSet
from ._physics.rng import component_rng
from ._physics.strikes import StrikeEvent, StruckResonatorPreset, render_strike_train
from .types import FloatAudio

_DOORBELL_NAMESPACE = 0x44424C4C


@dataclass(frozen=True, slots=True)
class DoorChimeModel:
    """Stable chime-bar identity and activation timing."""

    seed: int
    high_bar: StruckResonatorPreset
    low_bar: StruckResonatorPreset
    press_velocity_m_s: float
    release_velocity_m_s: float
    press_release_delay_s: float


def _bar_modes(fundamental_hz: float, seed: int, sample_rate: int) -> ModeSet:
    rng = component_rng(seed, _DOORBELL_NAMESPACE, round(fundamental_hz), sample_rate)
    modes = []
    for index, ratio in enumerate((1.0, 2.76, 5.4, 8.7)):
        frequency = fundamental_hz * ratio * float(rng.uniform(0.99, 1.01))
        if frequency < sample_rate * 0.44:
            modes.append(
                Mode(
                    frequency,
                    (0.72, 0.46, 0.27, 0.16)[index],
                    input_gain=(0.9, 0.54, 0.33, 0.18)[index],
                    radiation_gain=(0.8, 0.5, 0.3, 0.16)[index],
                    modal_mass_kg=0.02 + index * 0.006,
                )
            )
    return ModeSet(tuple(modes))


def generate_door_chime(*, seed: int, sample_rate: int) -> DoorChimeModel:
    """Generate deterministic high/low chime bars for one doorbell."""
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    rng = component_rng(seed, _DOORBELL_NAMESPACE, 1, sample_rate)
    contact = ImpactContact(0.0022, 8.5e7, exponent=1.5, restitution=0.28)
    high = StruckResonatorPreset(
        _bar_modes(float(rng.uniform(950.0, 1_150.0)), seed, sample_rate),
        contact,
        microscopic_gain=0.12,
        output_gain=0.82,
    )
    low = StruckResonatorPreset(
        _bar_modes(float(rng.uniform(680.0, 820.0)), seed, sample_rate),
        contact,
        microscopic_gain=0.12,
        output_gain=0.78,
    )
    return DoorChimeModel(
        seed=seed,
        high_bar=high,
        low_bar=low,
        press_velocity_m_s=float(rng.uniform(0.18, 0.27)),
        release_velocity_m_s=float(rng.uniform(0.15, 0.23)),
        press_release_delay_s=float(rng.uniform(0.15, 0.22)),
    )


def render_door_chime(
    model: DoorChimeModel,
    *,
    duration_s: float,
    sample_rate: int,
) -> FloatAudio:
    """Render the distinct high-then-low mechanical door-chime sequence."""
    events = (
        StrikeEvent(0.015, model.press_velocity_m_s, 0),
        StrikeEvent(model.press_release_delay_s, model.release_velocity_m_s, 1),
    )
    return render_strike_train(
        events=events,
        resonators=(model.high_bar, model.low_bar),
        duration_s=duration_s,
        sample_rate=sample_rate,
        rngs=(
            component_rng(model.seed, _DOORBELL_NAMESPACE, 2, sample_rate),
            component_rng(model.seed, _DOORBELL_NAMESPACE, 3, sample_rate),
        ),
    )


def render_electronic_doorbell(*, seed: int, duration_s: float, sample_rate: int) -> FloatAudio:
    """Render an electronic doorbell melody using the shared tone source."""
    preset = TonePreset(
        frequency_hz=1_040.0,
        harmonic_gains=(0.84, 0.22, 0.07),
        attack_s=0.006,
        release_s=0.18,
        bandwidth_hz=(500.0, 5_500.0),
        distortion=0.04,
        transient_click=0.0,
    )
    first_duration = min(0.24, duration_s)
    second_start = min(0.22, duration_s)
    pulses = [TonePulse(0.0, first_duration, level=0.7, pitch_scale=1.0)]
    if duration_s > second_start:
        pulses.append(
            TonePulse(
                second_start,
                min(0.34, duration_s - second_start),
                level=0.64,
                pitch_scale=1.25,
            )
        )
    return render_tone_pattern(
        preset=preset,
        pulses=pulses,
        duration_s=duration_s,
        sample_rate=sample_rate,
        rng=component_rng(seed, _DOORBELL_NAMESPACE, 10, sample_rate),
    )

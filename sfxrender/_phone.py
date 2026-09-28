"""Physical and electronic telephone ring sources."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._electronic import TonePreset, TonePulse, render_tone_pattern
from ._physics.contact import ImpactContact
from ._physics.modes import Mode, ModeSet
from ._physics.rng import component_rng
from ._physics.strikes import StrikeEvent, StruckResonatorPreset, render_strike_train
from .types import FloatAudio

_PHONE_NAMESPACE = 0x50484F4E


@dataclass(frozen=True, slots=True)
class ClassicPhoneRingerModel:
    """Stable generated identity for a two-gong telephone ringer."""

    seed: int
    gong_a: StruckResonatorPreset
    gong_b: StruckResonatorPreset
    housing_modes: ModeSet | None
    strike_rate_hz: float
    strike_velocity_m_s: float
    asymmetry: float


def _gong_modes(
    fundamental_hz: float,
    *,
    rng: np.random.Generator,
    sample_rate: int,
) -> ModeSet:
    ratios = (1.0, 1.43, 2.08, 2.91, 3.74)
    decays = (0.48, 0.39, 0.27, 0.19, 0.13)
    gains = (0.9, 0.72, 0.46, 0.31, 0.18)
    modes = tuple(
        Mode(
            frequency_hz=fundamental_hz * ratio * float(rng.uniform(0.992, 1.008)),
            decay_s=decay,
            input_gain=float(gain * rng.uniform(0.85, 1.15)),
            radiation_gain=gain,
            modal_mass_kg=0.012 + index * 0.004,
        )
        for index, (ratio, decay, gain) in enumerate(zip(ratios, decays, gains, strict=True))
        if fundamental_hz * ratio < sample_rate * 0.44
    )
    return ModeSet(modes)


def generate_classic_phone_ringer(
    *,
    seed: int,
    sample_rate: int,
) -> ClassicPhoneRingerModel:
    """Generate repeatable gong, housing, and clapper identity from one seed."""
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    identity_rng = component_rng(seed, _PHONE_NAMESPACE, 1, sample_rate)
    gong_a_frequency = float(identity_rng.uniform(430.0, 446.0))
    gong_b_frequency = float(identity_rng.uniform(468.0, 486.0))
    gong_a_modes = _gong_modes(
        gong_a_frequency,
        rng=component_rng(seed, _PHONE_NAMESPACE, 2, sample_rate),
        sample_rate=sample_rate,
    )
    gong_b_modes = _gong_modes(
        gong_b_frequency,
        rng=component_rng(seed, _PHONE_NAMESPACE, 3, sample_rate),
        sample_rate=sample_rate,
    )
    gong_a_contact = ImpactContact(0.0016, 1.4e8, exponent=1.5, restitution=0.32)
    gong_b_contact = ImpactContact(0.0018, 1.25e8, exponent=1.5, restitution=0.30)
    gong_a = StruckResonatorPreset(gong_a_modes, gong_a_contact, 0.18, 0.8)
    gong_b = StruckResonatorPreset(gong_b_modes, gong_b_contact, 0.18, 0.8)
    housing_modes = ModeSet(
        (
            Mode(238.0, 0.12, input_gain=0.5, radiation_gain=0.18, modal_mass_kg=0.08),
            Mode(372.0, 0.09, input_gain=0.34, radiation_gain=0.12, modal_mass_kg=0.06),
            Mode(820.0, 0.06, input_gain=0.22, radiation_gain=0.08, modal_mass_kg=0.04),
        )
    )
    return ClassicPhoneRingerModel(
        seed=seed,
        gong_a=gong_a,
        gong_b=gong_b,
        housing_modes=housing_modes,
        strike_rate_hz=float(identity_rng.uniform(19.0, 22.0)),
        strike_velocity_m_s=float(identity_rng.uniform(0.22, 0.31)),
        asymmetry=float(identity_rng.uniform(0.02, 0.10)),
    )


def render_classic_phone_ring(
    model: ClassicPhoneRingerModel,
    *,
    duration_s: float,
    sample_rate: int,
) -> FloatAudio:
    """Render a clapper strike train exciting stable telephone gongs."""
    resonators = [model.gong_a, model.gong_b]
    events: list[StrikeEvent] = []
    interval = 1.0 / model.strike_rate_hz
    strike_index = 0
    while strike_index * interval < duration_s:
        resonator_index = strike_index % 2
        asymmetry = 1.0 + model.asymmetry if resonator_index == 0 else 1.0 - model.asymmetry
        velocity = model.strike_velocity_m_s * asymmetry
        events.append(StrikeEvent(strike_index * interval, velocity, resonator_index))
        strike_index += 1

    if model.housing_modes is not None:
        housing = StruckResonatorPreset(
            model.housing_modes,
            model.gong_a.contact,
            microscopic_gain=0.0,
            output_gain=0.13,
        )
        resonators.append(housing)
        events.extend(
            StrikeEvent(event.time_s, event.velocity_m_s * 0.48, 2) for event in tuple(events)
        )

    return render_strike_train(
        events=events,
        resonators=resonators,
        duration_s=duration_s,
        sample_rate=sample_rate,
        rngs=tuple(
            component_rng(model.seed, _PHONE_NAMESPACE, 10 + index, sample_rate)
            for index in range(len(resonators))
        ),
    )


def render_electronic_phone_ring(*, seed: int, duration_s: float, sample_rate: int) -> FloatAudio:
    """Render a compact electronic telephone ring using the shared tone source."""
    preset = TonePreset(
        frequency_hz=880.0,
        harmonic_gains=(0.9, 0.34, 0.12),
        attack_s=0.008,
        release_s=0.025,
        bandwidth_hz=(320.0, 3_800.0),
        distortion=0.08,
        transient_click=0.025,
    )
    pulses = tuple(
        TonePulse(start_s=start, duration_s=min(0.105, duration_s - start), level=0.74)
        for start in np.arange(0.0, duration_s, 0.15)
        if duration_s - start > 0.0
    )
    return render_tone_pattern(
        preset=preset,
        pulses=pulses,
        duration_s=duration_s,
        sample_rate=sample_rate,
        rng=component_rng(seed, _PHONE_NAMESPACE, 20, sample_rate),
    )

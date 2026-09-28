"""Metallic keys, keyboard, clock, and alarm actions."""

from __future__ import annotations

import math

import numpy as np

from ._dsp import mix_at
from ._electronic import TonePreset, TonePulse, render_tone_pattern
from ._phone import generate_classic_phone_ringer, render_classic_phone_ring
from ._physics.contact import ImpactContact
from ._physics.modes import Mode, ModeSet
from ._physics.rng import component_rng
from ._physics.stochastic import sample_stochastic_events
from ._physics.strikes import StrikeEvent, StruckResonatorPreset, render_strike_train
from .types import FloatAudio

_METAL_NAMESPACE = 0x4D455441


def _resonator(
    specifications: tuple[tuple[float, float, float], ...],
    *,
    seed: int,
    sample_rate: int,
    component_id: int,
    contact: ImpactContact,
    output_gain: float,
) -> StruckResonatorPreset:
    rng = component_rng(seed, _METAL_NAMESPACE, component_id, sample_rate)
    modes = ModeSet(
        tuple(
            Mode(
                frequency * float(rng.uniform(0.985, 1.015)),
                decay,
                input_gain=gain * float(rng.uniform(0.9, 1.1)),
                radiation_gain=gain,
                modal_mass_kg=0.008 + index * 0.003,
            )
            for index, (frequency, decay, gain) in enumerate(specifications)
            if frequency < sample_rate * 0.43
        )
    )
    return StruckResonatorPreset(modes, contact, microscopic_gain=0.18, output_gain=output_gain)


def render_keys_jingle(*, sample_rate: int, duration: float, style: str, seed: int) -> FloatAudio:
    rng = component_rng(seed, _METAL_NAMESPACE, 1, sample_rate)
    contact = ImpactContact(0.0012, 4.5e7, exponent=1.5, restitution=0.58)
    resonators = (
        _resonator(
            (
                (780.0, 0.22, 0.8),
                (1_460.0, 0.16, 0.54),
                (2_650.0, 0.11, 0.30),
                (4_100.0, 0.08, 0.17),
            ),
            seed=seed,
            sample_rate=sample_rate,
            component_id=2,
            contact=contact,
            output_gain=3.2,
        ),
        _resonator(
            (
                (920.0, 0.17, 0.78),
                (1_880.0, 0.13, 0.48),
                (3_240.0, 0.09, 0.27),
                (4_760.0, 0.06, 0.14),
            ),
            seed=seed,
            sample_rate=sample_rate,
            component_id=3,
            contact=contact,
            output_gain=3.0,
        ),
        _resonator(
            (
                (610.0, 0.28, 0.8),
                (1_120.0, 0.21, 0.51),
                (2_180.0, 0.13, 0.28),
                (3_780.0, 0.08, 0.16),
            ),
            seed=seed,
            sample_rate=sample_rate,
            component_id=4,
            contact=contact,
            output_gain=3.4,
        ),
    )
    rate_hz = 18.0 if style == "full" else 8.0
    events = sample_stochastic_events(
        duration_s=duration,
        sample_rate=sample_rate,
        rate_hz=rate_hz,
        seed=seed,
        component_id=10,
        event_duration_s=(0.006, 0.022),
    )
    strikes = tuple(
        StrikeEvent(
            event.start_sample / sample_rate,
            float(rng.uniform(0.11, 0.28)) * event.level,
            int(rng.integers(0, len(resonators))),
        )
        for event in events
    )
    return render_strike_train(
        events=strikes,
        resonators=resonators,
        duration_s=duration,
        sample_rate=sample_rate,
        rngs=tuple(
            component_rng(seed, _METAL_NAMESPACE, 20 + index, sample_rate)
            for index in range(len(resonators))
        ),
    )


def render_clock_tick(
    *, sample_rate: int, duration: float, style: str, rate: str, seed: int
) -> FloatAudio:
    interval_s = 1.0 if rate == "slow" else 0.5
    contact = ImpactContact(0.001, 2.0e7, exponent=1.5, restitution=0.32)
    if style == "mantel":
        specifications = (
            ((320.0, 0.12, 0.62), (760.0, 0.09, 0.38), (1_520.0, 0.055, 0.20)),
            ((410.0, 0.10, 0.60), (890.0, 0.07, 0.36), (1_740.0, 0.05, 0.18)),
        )
    else:
        specifications = (
            ((680.0, 0.055, 0.68), (1_260.0, 0.042, 0.38), (2_100.0, 0.030, 0.18)),
            ((820.0, 0.050, 0.64), (1_540.0, 0.035, 0.34), (2_460.0, 0.025, 0.17)),
        )
    resonators = tuple(
        _resonator(
            specs,
            seed=seed,
            sample_rate=sample_rate,
            component_id=40 + index,
            contact=contact,
            output_gain=13.0,
        )
        for index, specs in enumerate(specifications)
    )
    events = tuple(
        StrikeEvent(index * interval_s, 0.045, index % 2)
        for index in range(math.ceil(duration / interval_s))
        if index * interval_s < duration
    )
    return render_strike_train(
        events=events,
        resonators=resonators,
        duration_s=duration,
        sample_rate=sample_rate,
        rngs=tuple(
            component_rng(seed, _METAL_NAMESPACE, 50 + index, sample_rate)
            for index in range(len(resonators))
        ),
    )


def _electronic_alarm(*, sample_rate: int, duration: float, pattern: str, seed: int) -> FloatAudio:
    preset = TonePreset(
        frequency_hz=1_040.0,
        harmonic_gains=(0.9, 0.22),
        attack_s=0.006,
        release_s=0.025,
        bandwidth_hz=(450.0, 3_800.0),
        distortion=0.04,
        transient_click=0.035,
    )
    pulse_period = 0.24 if pattern == "continuous" else 0.72
    pulse_duration = min(0.18, pulse_period * 0.82)
    pulses = tuple(
        TonePulse(start_s=start, duration_s=min(pulse_duration, duration - start), level=0.86)
        for start in np.arange(0.0, duration, pulse_period)
        if duration - start > 0.0
    )
    return render_tone_pattern(
        preset=preset,
        pulses=pulses,
        duration_s=duration,
        sample_rate=sample_rate,
        rng=component_rng(seed, _METAL_NAMESPACE, 60, sample_rate),
    )


def render_alarm_ring(
    *, sample_rate: int, duration: float, style: str, pattern: str, seed: int
) -> FloatAudio:
    output_size = round((duration + 0.35) * sample_rate)
    output = np.zeros(output_size, dtype=np.float32)
    if style == "electronic":
        mix_at(
            output,
            _electronic_alarm(
                sample_rate=sample_rate,
                duration=duration,
                pattern=pattern,
                seed=seed,
            ),
            0,
        )
        return output

    model = generate_classic_phone_ringer(seed=seed, sample_rate=sample_rate)
    burst_duration = duration if pattern == "continuous" else 0.82
    gap_s = 0.42
    start_s = 0.0
    while start_s < duration:
        active_duration = min(burst_duration, duration - start_s)
        ring = render_classic_phone_ring(
            model,
            duration_s=active_duration,
            sample_rate=sample_rate,
        )
        mix_at(output, ring, round(start_s * sample_rate))
        start_s += active_duration + (0.0 if pattern == "continuous" else gap_s)
    return output

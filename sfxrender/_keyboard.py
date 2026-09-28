"""Procedural computer-keyboard events and reusable struck resonators."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ._dsp import bandpass_noise, mix_at
from ._physics.contact import ImpactContact
from ._physics.modes import Mode, ModeSet
from ._physics.rng import component_rng
from ._physics.strikes import StrikeEvent, StruckResonatorPreset, render_strike_train
from .types import FloatAudio

_KEYBOARD_NAMESPACE = 0x4B455942
_SPEED_RATE_HZ = {"slow": 2.5, "steady": 5.5, "fast": 10.0}
_NORMAL_VARIANTS = 5
_SPECIAL_CLASSES = ("space", "enter", "backspace")


@dataclass(frozen=True, slots=True)
class KeyboardVoiceProfile:
    keycap_modes: ModeSet
    case_modes: ModeSet
    contact: ImpactContact
    press_gain: float
    bottom_out_gain: float
    release_gain: float
    case_coupling: float
    click_gain: float
    brightness: float
    hold_s: tuple[float, float]
    position_spread: float


@dataclass(frozen=True, slots=True)
class KeyEvent:
    press_time_s: float
    release_time_s: float
    key_class: str
    variant: int
    velocity: float
    position: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.press_time_s) or self.press_time_s < 0.0:
            raise ValueError("press_time_s must be finite and non-negative")
        if not math.isfinite(self.release_time_s) or self.release_time_s <= self.press_time_s:
            raise ValueError("release_time_s must be greater than press_time_s")
        if self.key_class not in {"normal", *_SPECIAL_CLASSES}:
            raise ValueError("key_class must be normal, space, enter, or backspace")
        if self.key_class == "normal" and not 0 <= self.variant < _NORMAL_VARIANTS:
            raise ValueError("normal key variant is out of range")
        if not math.isfinite(self.velocity) or self.velocity <= 0.0:
            raise ValueError("velocity must be finite and positive")
        if not math.isfinite(self.position) or not 0.0 <= self.position <= 1.0:
            raise ValueError("position must be between zero and one")


def _keyboard_rate(speed: str) -> float:
    return _SPEED_RATE_HZ[speed]


def _generate_typing_events(
    *, duration: float, speed: str, force: str, seed: int
) -> tuple[KeyEvent, ...]:
    """Generate a deterministic, bounded word-burst typing sequence."""
    rate_hz = _keyboard_rate(speed)
    if force not in {"light", "firm"}:
        raise ValueError("force must be light or firm")

    timing_rng = component_rng(seed, _KEYBOARD_NAMESPACE, 1)
    class_rng = component_rng(seed, _KEYBOARD_NAMESPACE, 2)
    variant_rng = component_rng(seed, _KEYBOARD_NAMESPACE, 3)
    velocity_rng = component_rng(seed, _KEYBOARD_NAMESPACE, 4)
    position_rng = component_rng(seed, _KEYBOARD_NAMESPACE, 5)
    hold_rng = component_rng(seed, _KEYBOARD_NAMESPACE, 6)
    variation = {"slow": 0.25, "steady": 0.20, "fast": 0.15}[speed]
    force_velocity = {"light": 0.115, "firm": 0.205}[force]
    hold_bounds = {"slow": (0.055, 0.16), "steady": (0.05, 0.135), "fast": (0.045, 0.105)}[speed]
    correction_probability = {"slow": 0.075, "steady": 0.06, "fast": 0.045}[speed]
    word_gap = {"slow": (0.26, 0.50), "steady": (0.18, 0.34), "fast": (0.12, 0.23)}[speed]

    def make_event(time_s: float, key_class: str) -> KeyEvent:
        variant = int(variant_rng.integers(0, _NORMAL_VARIANTS)) if key_class == "normal" else 0
        hold_s = float(hold_rng.uniform(*hold_bounds))
        velocity = force_velocity * float(velocity_rng.uniform(0.84, 1.16))
        return KeyEvent(
            press_time_s=time_s,
            release_time_s=time_s + hold_s,
            key_class=key_class,
            variant=variant,
            velocity=velocity,
            position=float(position_rng.uniform(0.0, 1.0)),
        )

    events: list[KeyEvent] = []
    time_s = float(timing_rng.uniform(0.025, 0.075))
    while time_s < duration:
        word_length = int(np.clip(round(timing_rng.normal(5.0, 1.8)), 2, 9))
        for _ in range(word_length):
            if time_s >= duration:
                break
            events.append(make_event(time_s, "normal"))
            interval = (1.0 / rate_hz) * float(
                np.clip(timing_rng.normal(1.0, variation), 0.58, 1.48)
            )
            time_s += interval
        if time_s >= duration:
            break

        if class_rng.random() < correction_probability:
            time_s += float(timing_rng.uniform(0.035, 0.095))
            if time_s < duration:
                events.append(make_event(time_s, "backspace"))
                time_s += float(timing_rng.uniform(0.11, 0.23))

        events.append(make_event(time_s, "space"))
        time_s += float(timing_rng.uniform(*word_gap))
        if class_rng.random() < 0.055:
            time_s += float(timing_rng.uniform(0.18, 0.48))
            if time_s < duration:
                events.append(make_event(time_s, "enter"))
                time_s += float(timing_rng.uniform(0.10, 0.20))

    return tuple(sorted(events, key=lambda event: event.press_time_s))


def _mode_set(
    specifications: tuple[tuple[float, float, float], ...],
    *,
    sample_rate: int,
    rng: np.random.Generator,
    frequency_spread: float,
    decay_spread: float,
    gain_scale: float,
) -> ModeSet:
    modes = tuple(
        Mode(
            frequency_hz=frequency
            * float(rng.uniform(1.0 - frequency_spread, 1.0 + frequency_spread)),
            decay_s=decay * float(rng.uniform(1.0 - decay_spread, 1.0 + decay_spread)),
            input_gain=gain * gain_scale * float(rng.uniform(0.84, 1.16)),
            radiation_gain=gain,
            modal_mass_kg=0.005 + index * 0.0025,
        )
        for index, (frequency, decay, gain) in enumerate(specifications)
        if frequency < sample_rate * 0.43
    )
    return ModeSet(modes)


def _build_keyboard_voices(*, sample_rate: int, seed: int) -> tuple[KeyboardVoiceProfile, ...]:
    """Build five related normal-key voices and three larger-key voices."""
    bases = {
        "normal": (
            ((780.0, 0.045, 0.66), (1_580.0, 0.034, 0.39), (2_680.0, 0.023, 0.20)),
            (
                (205.0, 0.15, 0.48),
                (430.0, 0.11, 0.32),
                (850.0, 0.075, 0.19),
                (1_420.0, 0.052, 0.10),
            ),
            ImpactContact(0.006, 1.4e7, exponent=1.5, restitution=0.20),
            (1.0, 0.64, 0.38, 0.28, 0.42, 0.82, (0.05, 0.135), 0.22),
        ),
        "space": (
            ((430.0, 0.085, 0.62), (860.0, 0.065, 0.36), (1_620.0, 0.045, 0.17)),
            ((118.0, 0.23, 0.68), (245.0, 0.19, 0.52), (485.0, 0.13, 0.30), (910.0, 0.085, 0.14)),
            ImpactContact(0.012, 0.95e7, exponent=1.5, restitution=0.16),
            (0.88, 0.58, 0.34, 0.58, 0.22, 0.47, (0.07, 0.16), 0.10),
        ),
        "enter": (
            ((610.0, 0.065, 0.65), (1_230.0, 0.048, 0.38), (2_120.0, 0.030, 0.19)),
            (
                (168.0, 0.19, 0.60),
                (350.0, 0.14, 0.42),
                (690.0, 0.095, 0.25),
                (1_180.0, 0.064, 0.13),
            ),
            ImpactContact(0.009, 1.15e7, exponent=1.5, restitution=0.18),
            (1.05, 0.72, 0.43, 0.48, 0.34, 0.68, (0.06, 0.15), 0.17),
        ),
        "backspace": (
            ((920.0, 0.038, 0.72), (1_840.0, 0.029, 0.44), (3_050.0, 0.019, 0.24)),
            (
                (235.0, 0.13, 0.40),
                (490.0, 0.095, 0.28),
                (990.0, 0.065, 0.17),
                (1_680.0, 0.044, 0.09),
            ),
            ImpactContact(0.007, 1.65e7, exponent=1.5, restitution=0.21),
            (1.12, 0.78, 0.47, 0.32, 0.54, 0.98, (0.05, 0.13), 0.20),
        ),
    }
    voices: list[KeyboardVoiceProfile] = []
    class_order = ("normal", *_SPECIAL_CLASSES)
    for key_class_index, key_class in enumerate(class_order):
        variants = range(_NORMAL_VARIANTS) if key_class == "normal" else range(1)
        key_specs, case_specs, contact, gains = bases[key_class]
        for variant in variants:
            variant_rng = component_rng(seed, _KEYBOARD_NAMESPACE, 100 + key_class_index, variant)
            frequency_scale = float(variant_rng.uniform(0.975, 1.025))
            damping_scale = float(variant_rng.uniform(0.90, 1.10))
            keycap_modes = _mode_set(
                key_specs,
                sample_rate=sample_rate,
                rng=component_rng(seed, _KEYBOARD_NAMESPACE, 200 + key_class_index, variant),
                frequency_spread=0.015,
                decay_spread=0.10,
                gain_scale=frequency_scale,
            )
            varied_keycap = ModeSet(
                tuple(
                    Mode(
                        mode.frequency_hz * frequency_scale,
                        mode.decay_s * damping_scale,
                        mode.input_gain,
                        mode.radiation_gain,
                        mode.modal_mass_kg,
                    )
                    for mode in keycap_modes.modes
                )
            )
            case_coupling, press, bottom, release, click, brightness, hold_s, position_spread = (
                gains
            )
            case_modes = _mode_set(
                case_specs,
                sample_rate=sample_rate,
                rng=component_rng(seed, _KEYBOARD_NAMESPACE, 300 + key_class_index, variant),
                frequency_spread=0.018,
                decay_spread=0.12,
                gain_scale=case_coupling,
            )
            voices.append(
                KeyboardVoiceProfile(
                    keycap_modes=varied_keycap,
                    case_modes=case_modes,
                    contact=contact,
                    press_gain=press,
                    bottom_out_gain=bottom,
                    release_gain=release,
                    case_coupling=case_coupling,
                    click_gain=click,
                    brightness=brightness,
                    hold_s=hold_s,
                    position_spread=position_spread,
                )
            )
    return tuple(voices)


def _voice_index(event: KeyEvent) -> int:
    if event.key_class == "normal":
        return event.variant
    return _NORMAL_VARIANTS + _SPECIAL_CLASSES.index(event.key_class)


def _render_keyboard_events(
    events: tuple[KeyEvent, ...], *, duration: float, sample_rate: int, seed: int
) -> FloatAudio:
    voices = _build_keyboard_voices(sample_rate=sample_rate, seed=seed)
    resonators: list[StruckResonatorPreset] = []
    for voice in voices:
        resonators.append(
            StruckResonatorPreset(
                voice.keycap_modes,
                voice.contact,
                microscopic_gain=0.14,
                output_gain=2.8,
            )
        )
        resonators.append(
            StruckResonatorPreset(
                voice.case_modes,
                voice.contact,
                microscopic_gain=0.12,
                output_gain=2.2,
            )
        )

    strikes: list[StrikeEvent] = []
    for event_index, event in enumerate(events):
        voice = voices[_voice_index(event)]
        keycap_index = 2 * _voice_index(event)
        case_index = keycap_index + 1
        event_rng = component_rng(seed, _KEYBOARD_NAMESPACE, 400, event_index)
        position_gain = 1.0 + voice.position_spread * (0.5 - event.position)
        velocity = event.velocity
        bottom_delay = float(event_rng.uniform(0.003, 0.012))
        release_delay = float(event_rng.uniform(0.001, 0.004))
        strikes.extend(
            (
                StrikeEvent(event.press_time_s, velocity * voice.press_gain, keycap_index),
                StrikeEvent(
                    event.press_time_s + bottom_delay,
                    velocity * voice.bottom_out_gain * position_gain,
                    case_index,
                ),
                StrikeEvent(event.release_time_s, velocity * voice.release_gain, keycap_index),
                StrikeEvent(
                    event.release_time_s + release_delay,
                    velocity * voice.release_gain * voice.case_coupling * position_gain,
                    case_index,
                ),
            )
        )

    render_duration = duration + 0.22
    output = render_strike_train(
        events=strikes,
        resonators=resonators,
        duration_s=render_duration,
        sample_rate=sample_rate,
        rngs=tuple(
            component_rng(seed, _KEYBOARD_NAMESPACE, 500 + index, sample_rate)
            for index in range(len(resonators))
        ),
    )
    for event_index, event in enumerate(events):
        voice = voices[_voice_index(event)]
        size = max(1, round(0.0045 * sample_rate))
        low_hz = 700.0 + 900.0 * voice.brightness
        high_hz = min(6_200.0, sample_rate * 0.46)
        noise = bandpass_noise(
            component_rng(seed, _KEYBOARD_NAMESPACE, 600, event_index),
            size,
            sample_rate,
            low_hz,
            high_hz,
        )
        envelope = np.exp(-np.arange(size, dtype=np.float32) / np.float32(0.0016 * sample_rate))
        click = noise * envelope * np.float32(0.018 * voice.click_gain * event.velocity)
        mix_at(output, np.asarray(click, dtype=np.float32), round(event.press_time_s * sample_rate))
    return output


def render_keyboard_typing(
    *, sample_rate: int, duration: float, speed: str, force: str, seed: int
) -> FloatAudio:
    events = _generate_typing_events(duration=duration, speed=speed, force=force, seed=seed)
    return _render_keyboard_events(events, duration=duration, sample_rate=sample_rate, seed=seed)

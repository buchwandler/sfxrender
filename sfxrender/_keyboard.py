"""Procedural computer-keyboard events and reusable struck resonators."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ._dsp import asymmetric_pulse, bandpass_noise, mix_at, one_pole_highpass
from ._physics.contact import ImpactContact, impact_force
from ._physics.modes import Mode, ModeSet
from ._physics.physical_impacts import render_force_response
from ._physics.rng import component_rng
from .types import FloatAudio

_KEYBOARD_NAMESPACE = 0x4B455942
_SPEED_RATE_HZ = {"slow": 2.5, "steady": 5.5, "fast": 10.0}
_NORMAL_VARIANTS = 5
_SPECIAL_CLASSES = ("space", "enter", "backspace")


FrequencyBand = tuple[float, float]
BandPair = tuple[FrequencyBand, FrequencyBand]


@dataclass(frozen=True, slots=True)
class KeyboardAcousticProfile:
    press_bands_hz: BandPair
    press_duration_s: float
    press_decay_s: float
    bottom_bands_hz: BandPair
    bottom_duration_s: float
    bottom_decay_s: float
    bottom_delay_s: tuple[float, float]
    release_bands_hz: BandPair
    release_duration_s: float
    release_decay_s: float
    release_gain: float
    body_modes: ModeSet
    body_gain: float
    body_tail_s: float
    press_gain: float
    bottom_gain: float
    stabilizer_gain: float
    stabilizer_count: tuple[int, int]
    brightness: float
    position_spread: float
    contact: ImpactContact


@dataclass(frozen=True, slots=True)
class _KeyboardProfileDesign:
    press_bands_hz: BandPair
    press_duration_s: float
    press_decay_s: float
    bottom_bands_hz: BandPair
    bottom_duration_s: float
    bottom_decay_s: float
    bottom_delay_s: tuple[float, float]
    release_bands_hz: BandPair
    release_duration_s: float
    release_decay_s: float
    release_gain: float
    mode_frequency_ranges_hz: tuple[FrequencyBand, ...]
    mode_decay_ranges_s: tuple[FrequencyBand, ...]
    body_gain: float
    body_tail_s: float
    press_gain: float
    bottom_gain: float
    stabilizer_gain: float
    stabilizer_count: tuple[int, int]
    position_spread: float
    contact: ImpactContact



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

    recent_variants: list[int] = []
    def make_event(time_s: float, key_class: str) -> KeyEvent:
        if key_class == "normal":
            variant = int(variant_rng.integers(0, _NORMAL_VARIANTS))
            available = [candidate for candidate in range(_NORMAL_VARIANTS) if candidate not in recent_variants]
            if variant in recent_variants and available:
                variant = available[int(variant_rng.integers(0, len(available)))]
            recent_variants.append(variant)
            del recent_variants[:-3]
        else:
            variant = 0
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


_KEYBOARD_PROFILE_DESIGNS: dict[str, _KeyboardProfileDesign] = {
    "normal": _KeyboardProfileDesign(
        press_bands_hz=((250.0, 1_800.0), (1_500.0, 6_500.0)),
        press_duration_s=0.012,
        press_decay_s=0.0032,
        bottom_bands_hz=((180.0, 1_300.0), (1_200.0, 4_500.0)),
        bottom_duration_s=0.014,
        bottom_decay_s=0.0040,
        bottom_delay_s=(0.003, 0.009),
        release_bands_hz=((1_800.0, 3_800.0), (3_300.0, 7_000.0)),
        release_duration_s=0.009,
        release_decay_s=0.0025,
        release_gain=0.30,
        mode_frequency_ranges_hz=((165.0, 230.0), (330.0, 470.0), (610.0, 870.0), (1_150.0, 1_750.0)),
        mode_decay_ranges_s=((0.035, 0.070), (0.025, 0.055), (0.018, 0.045), (0.012, 0.030)),
        body_gain=0.72,
        body_tail_s=0.085,
        press_gain=0.15,
        bottom_gain=0.09,
        stabilizer_gain=0.0,
        stabilizer_count=(0, 0),
        position_spread=0.12,
        contact=ImpactContact(0.006, 1.4e7, exponent=1.5, restitution=0.20),
    ),
    "space": _KeyboardProfileDesign(
        press_bands_hz=((180.0, 1_250.0), (1_000.0, 4_800.0)),
        press_duration_s=0.014,
        press_decay_s=0.0040,
        bottom_bands_hz=((120.0, 900.0), (750.0, 3_200.0)),
        bottom_duration_s=0.017,
        bottom_decay_s=0.0055,
        bottom_delay_s=(0.005, 0.013),
        release_bands_hz=((900.0, 2_400.0), (2_100.0, 6_000.0)),
        release_duration_s=0.010,
        release_decay_s=0.0030,
        release_gain=0.22,
        mode_frequency_ranges_hz=((115.0, 195.0), (240.0, 390.0), (470.0, 760.0), (850.0, 1_450.0)),
        mode_decay_ranges_s=((0.055, 0.140), (0.050, 0.120), (0.035, 0.090), (0.020, 0.060)),
        body_gain=1.20,
        body_tail_s=0.145,
        press_gain=0.14,
        bottom_gain=0.13,
        stabilizer_gain=0.025,
        stabilizer_count=(1, 3),
        position_spread=0.20,
        contact=ImpactContact(0.012, 0.95e7, exponent=1.5, restitution=0.16),
    ),
    "enter": _KeyboardProfileDesign(
        press_bands_hz=((220.0, 1_500.0), (1_400.0, 5_800.0)),
        press_duration_s=0.013,
        press_decay_s=0.0036,
        bottom_bands_hz=((150.0, 1_050.0), (950.0, 3_800.0)),
        bottom_duration_s=0.015,
        bottom_decay_s=0.0048,
        bottom_delay_s=(0.005, 0.012),
        release_bands_hz=((1_000.0, 2_700.0), (2_400.0, 6_500.0)),
        release_duration_s=0.009,
        release_decay_s=0.0027,
        release_gain=0.22,
        mode_frequency_ranges_hz=((150.0, 225.0), (300.0, 465.0), (560.0, 850.0), (1_000.0, 1_650.0)),
        mode_decay_ranges_s=((0.045, 0.110), (0.035, 0.090), (0.025, 0.065), (0.018, 0.045)),
        body_gain=0.92,
        body_tail_s=0.120,
        press_gain=0.15,
        bottom_gain=0.11,
        stabilizer_gain=0.022,
        stabilizer_count=(1, 2),
        position_spread=0.17,
        contact=ImpactContact(0.009, 1.15e7, exponent=1.5, restitution=0.18),
    ),
    "backspace": _KeyboardProfileDesign(
        press_bands_hz=((330.0, 2_100.0), (2_000.0, 7_000.0)),
        press_duration_s=0.011,
        press_decay_s=0.0028,
        bottom_bands_hz=((220.0, 1_400.0), (1_500.0, 5_300.0)),
        bottom_duration_s=0.012,
        bottom_decay_s=0.0035,
        bottom_delay_s=(0.003, 0.008),
        release_bands_hz=((1_400.0, 3_500.0), (3_000.0, 7_000.0)),
        release_duration_s=0.008,
        release_decay_s=0.0022,
        release_gain=0.25,
        mode_frequency_ranges_hz=((190.0, 280.0), (410.0, 620.0), (750.0, 1_150.0), (1_400.0, 2_200.0)),
        mode_decay_ranges_s=((0.030, 0.075), (0.025, 0.060), (0.018, 0.045), (0.012, 0.030)),
        body_gain=0.62,
        body_tail_s=0.080,
        press_gain=0.16,
        bottom_gain=0.085,
        stabilizer_gain=0.0,
        stabilizer_count=(0, 0),
        position_spread=0.18,
        contact=ImpactContact(0.007, 1.65e7, exponent=1.5, restitution=0.21),
    ),
}


def _safe_band(sample_rate: int, low_hz: float, high_hz: float) -> FrequencyBand:
    """Clamp a requested noise band below Nyquist without collapsing its width."""
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    nyquist_margin = sample_rate * 0.46
    minimum_width = min(nyquist_margin * 0.20, max(80.0, sample_rate * 0.025))
    high = min(max(float(high_hz), 1.0), nyquist_margin)
    low = min(max(float(low_hz), 1.0), max(1.0, high - minimum_width))
    if high - low < 1.0:
        low = max(1.0, high - minimum_width)
    return low, high


def _keyboard_transient(
    *,
    sample_rate: int,
    rng: np.random.Generator,
    duration_s: float,
    bands_hz: BandPair,
    attack_s: float,
    decay_s: float,
    gain: float,
    brightness: float,
    body_weight: float = 0.48,
    presence_weight: float = 0.68,
) -> FloatAudio:
    """Synthesize a compact two-band, aperiodic mechanical contact."""
    size = max(1, round(duration_s * sample_rate))
    body_low, body_high = _safe_band(sample_rate, *bands_hz[0])
    presence_low, presence_high = _safe_band(sample_rate, *bands_hz[1])
    body = bandpass_noise(rng, size, sample_rate, body_low, body_high)
    presence = bandpass_noise(rng, size, sample_rate, presence_low, presence_high)
    envelope = asymmetric_pulse(
        size, sample_rate, attack_s=attack_s, decay_s=decay_s
    )
    envelope_peak = float(np.max(envelope)) if envelope.size else 0.0
    if envelope_peak > 0.0:
        envelope /= np.float32(envelope_peak)
    pressure = asymmetric_pulse(
        size,
        sample_rate,
        attack_s=max(1.0 / sample_rate, attack_s * 0.35),
        decay_s=max(2.0 / sample_rate, decay_s * 0.12),
        amplitude=0.18,
    )
    pressure = one_pole_highpass(pressure, sample_rate, max(120.0, body_low))
    texture = np.float32(body_weight) * body + np.float32(presence_weight * brightness) * presence
    return np.asarray(
        (texture * envelope + pressure) * np.float32(gain), dtype=np.float32
    )


def _build_body_modes(
    *,
    sample_rate: int,
    rng: np.random.Generator,
    design: _KeyboardProfileDesign,
 ) -> ModeSet:
    modes = tuple(
        Mode(
            frequency_hz=float(rng.uniform(*frequency_range)),
            decay_s=float(rng.uniform(*decay_range)),
            input_gain=float(rng.uniform(0.72, 0.96)),
            radiation_gain=float(rng.uniform(0.22, 0.36)),
            modal_mass_kg=0.008 + index * 0.0025,
        )
        for index, (frequency_range, decay_range) in enumerate(
            zip(
                design.mode_frequency_ranges_hz,
                design.mode_decay_ranges_s,
                strict=True,
            )
        )
        if frequency_range[0] < sample_rate * 0.43
    )
    return ModeSet(modes)


def _scale_mode_decays(modes: ModeSet, scale: float) -> ModeSet:
    """Change modal damping while preserving generated keyboard frequencies."""
    return ModeSet(
        tuple(
            Mode(
                mode.frequency_hz,
                mode.decay_s * scale,
                mode.input_gain,
                mode.radiation_gain,
                mode.modal_mass_kg,
            )
            for mode in modes.modes
        )
    )


def _build_keyboard_voices(
    *, sample_rate: int, seed: int
 ) -> tuple[KeyboardAcousticProfile, ...]:
    """Build one stable, irregular body identity with related key profiles."""
    identity_rng = component_rng(seed, _KEYBOARD_NAMESPACE, 90, sample_rate)
    global_brightness = float(identity_rng.uniform(0.97, 1.03))
    brightness_by_class = {"normal": 1.0, "space": 0.84, "enter": 0.98, "backspace": 1.12}
    profiles: list[KeyboardAcousticProfile] = []
    class_order = ("normal", *_SPECIAL_CLASSES)
    for class_index, key_class in enumerate(class_order):
        design = _KEYBOARD_PROFILE_DESIGNS[key_class]
        body_modes = _build_body_modes(
            sample_rate=sample_rate,
            rng=component_rng(seed, _KEYBOARD_NAMESPACE, 100 + class_index, sample_rate),
            design=design,
        )
        variants = range(_NORMAL_VARIANTS) if key_class == "normal" else range(1)
        for variant in variants:
            variation_rng = component_rng(
                seed,
                _KEYBOARD_NAMESPACE,
                150 + class_index,
                sample_rate * _NORMAL_VARIANTS + variant,
            )
            brightness = global_brightness * brightness_by_class[key_class]
            if key_class == "normal":
                brightness *= float(variation_rng.uniform(0.94, 1.06))
            else:
                brightness *= float(variation_rng.uniform(0.98, 1.02))
            profiles.append(
                KeyboardAcousticProfile(
                    press_bands_hz=design.press_bands_hz,
                    press_duration_s=design.press_duration_s,
                    press_decay_s=design.press_decay_s,
                    bottom_bands_hz=design.bottom_bands_hz,
                    bottom_duration_s=design.bottom_duration_s,
                    bottom_decay_s=design.bottom_decay_s,
                    bottom_delay_s=design.bottom_delay_s,
                    release_bands_hz=design.release_bands_hz,
                    release_duration_s=design.release_duration_s,
                    release_decay_s=design.release_decay_s,
                    release_gain=design.release_gain,
                    body_modes=body_modes,
                    body_gain=design.body_gain,
                    body_tail_s=design.body_tail_s,
                    press_gain=design.press_gain
                    * float(variation_rng.uniform(0.97, 1.03)),
                    bottom_gain=design.bottom_gain,
                    stabilizer_gain=design.stabilizer_gain,
                    stabilizer_count=design.stabilizer_count,
                    brightness=brightness,
                    position_spread=design.position_spread,
                    contact=design.contact,
                )
            )
    return tuple(profiles)


def _voice_index(event: KeyEvent) -> int:
    if event.key_class == "normal":
        return event.variant
    return _NORMAL_VARIANTS + _SPECIAL_CLASSES.index(event.key_class)


def _render_keyboard_events(
    events: tuple[KeyEvent, ...],
    *,
    duration: float,
    sample_rate: int,
    seed: int,
    speed: str = "steady",
) -> FloatAudio:
    """Render contacts with speed-aware damping and seed-stable body frequencies."""
    profiles = _build_keyboard_voices(sample_rate=sample_rate, seed=seed)
    body_decay_scale = {"slow": 1.05, "steady": 1.0, "fast": 0.88}[speed]
    release_level_scale = {"slow": 1.05, "steady": 1.0, "fast": 0.88}[speed]
    brightness_scale = {"slow": 0.98, "steady": 1.0, "fast": 1.04}[speed]
    body_level_scale = {"slow": 1.0, "steady": 1.0, "fast": 0.90}[speed]
    speed_modes = tuple(
        _scale_mode_decays(profile.body_modes, body_decay_scale) for profile in profiles
    )
    output = np.zeros(max(1, round((duration + 0.22) * sample_rate)), dtype=np.float32)
    for event_index, event in enumerate(events):
        profile_index = _voice_index(event)
        profile = profiles[profile_index]
        velocity_scale = event.velocity / 0.115
        position_gain = 1.0 + profile.position_spread * (0.5 - event.position)
        force_brightness = 1.0 + 0.04 * (velocity_scale - 1.0)
        event_brightness = profile.brightness * brightness_scale * force_brightness
        timing_rng = component_rng(seed, _KEYBOARD_NAMESPACE, 401, event_index)
        bottom_delay = float(timing_rng.uniform(*profile.bottom_delay_s))
        press_rng = component_rng(seed, _KEYBOARD_NAMESPACE, 410, event_index)
        press_gain_variation = float(press_rng.uniform(0.94, 1.06))
        press_decay_variation = float(press_rng.uniform(0.92, 1.08))
        press = _keyboard_transient(
            sample_rate=sample_rate,
            rng=press_rng,
            duration_s=profile.press_duration_s,
            bands_hz=profile.press_bands_hz,
            attack_s=0.00022,
            decay_s=profile.press_decay_s * press_decay_variation,
            gain=profile.press_gain * velocity_scale * press_gain_variation,
            brightness=event_brightness,
        )
        press_start = round(event.press_time_s * sample_rate)
        mix_at(output, press, press_start)

        bottom_rng = component_rng(seed, _KEYBOARD_NAMESPACE, 420, event_index)
        bottom_gain_variation = float(bottom_rng.uniform(0.92, 1.08))
        bottom_decay_variation = float(bottom_rng.uniform(0.92, 1.08))
        bottom = _keyboard_transient(
            sample_rate=sample_rate,
            rng=bottom_rng,
            duration_s=profile.bottom_duration_s,
            bands_hz=profile.bottom_bands_hz,
            attack_s=0.00025,
            decay_s=profile.bottom_decay_s * bottom_decay_variation,
            gain=profile.bottom_gain
            * velocity_scale
            * position_gain
            * bottom_gain_variation,
            brightness=event_brightness * 0.82,
        )
        bottom_start = press_start + round(bottom_delay * sample_rate)
        mix_at(output, bottom, bottom_start)

        trace = impact_force(
            contact=profile.contact,
            velocity_m_s=event.velocity,
            sample_rate=sample_rate,
        )
        body_coupling = 0.90 + 0.10 * velocity_scale
        body = render_force_response(
            trace.force_n,
            speed_modes[profile_index],
            sample_rate=sample_rate,
            rng=component_rng(seed, _KEYBOARD_NAMESPACE, 430, event_index),
            microscopic_gain=0.025,
            output_gain=profile.body_gain
            * position_gain
            * body_coupling
            * body_level_scale,
            tail_s=profile.body_tail_s,
        )
        mix_at(output, body, bottom_start)

        release_rng = component_rng(seed, _KEYBOARD_NAMESPACE, 440, event_index)
        release_gain_variation = float(release_rng.uniform(0.90, 1.10))
        release = _keyboard_transient(
            sample_rate=sample_rate,
            rng=release_rng,
            duration_s=profile.release_duration_s,
            bands_hz=profile.release_bands_hz,
            attack_s=0.00012,
            decay_s=profile.release_decay_s * press_decay_variation,
            gain=profile.press_gain
            * profile.release_gain
            * velocity_scale
            * release_level_scale
            * release_gain_variation,
            brightness=event_brightness * 1.16,
            body_weight=0.18,
            presence_weight=1.25,
        )
        mix_at(output, release, round(event.release_time_s * sample_rate))

        if profile.stabilizer_count[1] > 0:
            stabilizer_rng = component_rng(
                seed, _KEYBOARD_NAMESPACE, 450, event_index
            )
            count = int(
                stabilizer_rng.integers(
                    profile.stabilizer_count[0], profile.stabilizer_count[1] + 1
                )
            )
            for contact_index in range(count):
                delay = bottom_delay + float(stabilizer_rng.uniform(0.0005, 0.0040))
                contact_duration = float(stabilizer_rng.uniform(0.001, 0.004))
                contact_gain = profile.stabilizer_gain * velocity_scale * float(
                    stabilizer_rng.uniform(0.75, 1.15)
                )
                rattle = _keyboard_transient(
                    sample_rate=sample_rate,
                    rng=component_rng(
                        seed,
                        _KEYBOARD_NAMESPACE,
                        460 + contact_index,
                        event_index,
                    ),
                    duration_s=contact_duration,
                    bands_hz=((1_500.0, 3_200.0), (3_000.0, 7_000.0)),
                    attack_s=0.00008,
                    decay_s=0.0014,
                    gain=contact_gain,
                    brightness=event_brightness * 1.18,
                )
                mix_at(output, rattle, press_start + round(delay * sample_rate))
    return output


def render_keyboard_typing(
    *, sample_rate: int, duration: float, speed: str, force: str, seed: int
) -> FloatAudio:
    events = _generate_typing_events(duration=duration, speed=speed, force=force, seed=seed)
    return _render_keyboard_events(
        events, duration=duration, sample_rate=sample_rate, seed=seed, speed=speed
    )

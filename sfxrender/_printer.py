"""Shared deterministic synthesis layers for small electromechanical printers."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ._dsp import (
    asymmetric_pulse,
    bandpass_noise,
    envelope_from_points,
    mix_at,
    modal_bank,
    one_pole_highpass,
    one_pole_lowpass,
)
from ._physics.motion import minimum_jerk_motion
from .types import FloatAudio
from ._physics.contact import ImpactContact
from ._physics.modes import Mode, ModeSet
from ._physics.rng import component_rng
from ._electronic import TonePreset, TonePulse, render_tone_pattern
from ._physics.electromechanical import render_electromechanical_hum
from ._physics.rotating import render_rotating_machine
from ._physics.closure import render_terminal_closure
from ._physics.models import FrictionProfile, ModalBody, Mode as SlidingMode, MotionCurve
from ._physics.sliding import render_sliding_source

_PRINTER_SALT = 0x50524E54
_SWITCH_CONTACT = 1
_MOTOR = 2
_GEAR = 3
_PAPER = 4
_FAN = 5
_CHASSIS = 6
_VARIATION = 7
_TONE = 12
_CHASSIS_MODES = (
    (185.0, 0.080, 0.34),
    (315.0, 0.065, 0.30),
    (540.0, 0.050, 0.25),
    (920.0, 0.038, 0.20),
    (1540.0, 0.028, 0.14),
    (2380.0, 0.020, 0.08),
)
_PAGE_DURATION = {"slow": 1.30, "normal": 1.00, "fast": 0.78}
_TRAY_DURATION = {"slow": 0.78, "normal": 0.55, "fast": 0.38}
_RESTART_DURATION = {"slow": 4.0, "normal": 2.8, "fast": 1.9}
_WAKE_DURATION = {"light": 0.70, "deep": 1.15}


@dataclass(frozen=True, slots=True)
class PrinterModel:
    """Stable generated chassis, tray, motor, fan, and contact identity."""

    seed: int
    sample_rate: int
    chassis_modes: ModeSet
    tray_modes: ModeSet
    relay_contact: ImpactContact
    switch_contact: ImpactContact
    tray_stop_contact: ImpactContact
    latch_contact: ImpactContact
    motor_nominal_hz: float
    motor_detune_hz: float
    motor_wobble_hz: float
    motor_ripple_rate_hz: float
    fan_band_hz: tuple[float, float]


def generate_printer_model(*, seed: int, sample_rate: int) -> PrinterModel:
    """Generate deterministic printer identity for a seed and sample rate."""
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    chassis_rng = component_rng(seed, _PRINTER_SALT, _CHASSIS, sample_rate)
    chassis_modes = ModeSet(
        tuple(
            Mode(
                frequency * float(chassis_rng.uniform(0.985, 1.015)),
                decay * float(chassis_rng.uniform(0.94, 1.06)),
                input_gain=float(gain * chassis_rng.uniform(0.88, 1.12)),
                radiation_gain=gain,
                modal_mass_kg=0.12 + index * 0.04,
            )
            for index, (frequency, decay, gain) in enumerate(_CHASSIS_MODES)
            if frequency < sample_rate * 0.44
        )
    )
    tray_rng = component_rng(seed, _PRINTER_SALT, 8, sample_rate)
    tray_modes = ModeSet(
        tuple(
            Mode(
                frequency * float(tray_rng.uniform(0.97, 1.03)),
                decay * float(tray_rng.uniform(0.92, 1.08)),
                input_gain=float(gain * tray_rng.uniform(0.85, 1.15)),
                radiation_gain=gain,
                modal_mass_kg=0.06 + index * 0.025,
            )
            for index, (frequency, decay, gain) in enumerate(
                ((240.0, 0.12, 0.36), (480.0, 0.09, 0.30), (860.0, 0.065, 0.24),
                 (1_430.0, 0.045, 0.18), (2_250.0, 0.03, 0.11))
            )
            if frequency < sample_rate * 0.44
        )
    )

    def contact(component_id: int, mass: tuple[float, float], stiffness: tuple[float, float]) -> ImpactContact:
        rng = component_rng(seed, _PRINTER_SALT, component_id, sample_rate)
        return ImpactContact(
            effective_mass_kg=float(rng.uniform(*mass)),
            stiffness=float(rng.uniform(*stiffness)),
            exponent=1.5,
            restitution=float(rng.uniform(0.25, 0.48)),
        )

    motor_rng = component_rng(seed, _PRINTER_SALT, _MOTOR, sample_rate)
    fan_rng = component_rng(seed, _PRINTER_SALT, _FAN, sample_rate)
    return PrinterModel(
        seed=seed,
        sample_rate=sample_rate,
        chassis_modes=chassis_modes,
        tray_modes=tray_modes,
        relay_contact=contact(_SWITCH_CONTACT, (0.001, 0.004), (6.0e7, 1.4e8)),
        switch_contact=contact(9, (0.001, 0.003), (8.0e7, 1.8e8)),
        tray_stop_contact=contact(10, (0.015, 0.05), (1.2e7, 3.6e7)),
        latch_contact=contact(11, (0.001, 0.006), (4.0e7, 1.1e8)),
        motor_nominal_hz=float(motor_rng.uniform(76.0, 112.0)),
        motor_detune_hz=float(motor_rng.uniform(-1.8, 1.8)),
        motor_wobble_hz=float(motor_rng.uniform(1.1, 2.8)),
        motor_ripple_rate_hz=float(motor_rng.uniform(8.0, 19.0)),
        fan_band_hz=(float(fan_rng.uniform(75.0, 110.0)), float(fan_rng.uniform(3_000.0, 4_000.0))),
    )


def _resolve_model(seed: int, sample_rate: int, model: PrinterModel | None) -> PrinterModel:
    if model is None:
        return generate_printer_model(seed=seed, sample_rate=sample_rate)
    if model.seed != seed or model.sample_rate != sample_rate:
        raise ValueError("PrinterModel identity must match seed and sample_rate")
    return model


def _component_rng(seed: int, event_index: int, component_id: int) -> np.random.Generator:
    """Make an independent, stable stream for one physical layer and event."""
    sequence = np.random.SeedSequence([int(seed), _PRINTER_SALT, event_index, component_id])
    return np.random.default_rng(sequence)


def _click(
    sample_rate: int,
    seed: int,
    event_index: int,
    *,
    strength: float = 1.0,
    model: PrinterModel | None = None,
    body: str = "chassis",
) -> FloatAudio:
    model = _resolve_model(seed, sample_rate, model)
    size = max(1, round(0.105 * sample_rate))
    pulse = asymmetric_pulse(
        size,
        sample_rate,
        attack_s=0.0015,
        decay_s=0.022,
        amplitude=0.72,
    )
    rng = _component_rng(seed, event_index, _SWITCH_CONTACT)
    burst = bandpass_noise(rng, size, sample_rate, 180.0, min(6_500.0, sample_rate * 0.44))
    burst_time = np.arange(size, dtype=np.float32) / np.float32(sample_rate)
    burst *= np.exp(-burst_time / np.float32(0.012))
    excitation = np.asarray(pulse * 0.72 + burst * 0.20, dtype=np.float32)
    mode_set = model.chassis_modes if body == "chassis" else model.tray_modes
    mode_rng = _component_rng(model.seed, 0, _CHASSIS if body == "chassis" else _GEAR)
    modes = tuple(
        (mode.frequency_hz, mode.decay_s, mode.input_gain * mode.radiation_gain)
        for mode in mode_set.modes
        if mode.frequency_hz < sample_rate * 0.46
    )
    body_response = modal_bank(excitation, sample_rate, mode_rng, modes)
    return np.asarray(
        (excitation * 0.58 + body_response * 0.24) * np.float32(strength), dtype=np.float32
    )


def _motor_whir(
    sample_rate: int,
    size: int,
    seed: int,
    event_index: int,
    *,
    amplitude: float = 0.22,
    envelope: FloatAudio | None = None,
    model: PrinterModel | None = None,
) -> FloatAudio:
    if size <= 0:
        return np.zeros(0, dtype=np.float32)
    model = _resolve_model(seed, sample_rate, model)
    variation = _component_rng(seed, event_index, _VARIATION)
    time = np.arange(size, dtype=np.float64) / sample_rate
    duration = size / sample_rate
    base_hz = model.motor_nominal_hz
    attack = max(0.025, min(0.14, duration * 0.24))
    ramp = np.clip(time / attack, 0.0, 1.0)
    release_s = max(0.045, min(0.18, duration * 0.26))
    release = np.clip((duration - time) / release_s, 0.0, 1.0)
    motion = ramp * release
    frequency = base_hz * (0.78 + 0.22 * ramp) + model.motor_detune_hz
    frequency += 1.1 * np.sin(2.0 * math.pi * model.motor_wobble_hz * time)
    tone = render_rotating_machine(
        speed_curve_hz=np.asarray(frequency, dtype=np.float32),
        sample_rate=sample_rate,
        seed=seed,
        event_index=event_index,
        ripple_rate_hz=model.motor_ripple_rate_hz,
        amplitude=0.21,
    )
    texture = bandpass_noise(
        variation,
        size,
        sample_rate,
        120.0,
        min(2_200.0, sample_rate * 0.44),
    )
    if envelope is None:
        env = np.asarray(motion, dtype=np.float32)
    else:
        env = np.asarray(envelope[:size], dtype=np.float32) * np.asarray(motion, dtype=np.float32)
    signal = (tone.astype(np.float64) + texture.astype(np.float64) * 0.11) * env
    return np.asarray(signal * amplitude, dtype=np.float32)


def _fan_noise(
    sample_rate: int,
    size: int,
    seed: int,
    event_index: int,
    *,
    amplitude: float = 0.055,
    envelope: FloatAudio | None = None,
    model: PrinterModel | None = None,
) -> FloatAudio:
    if size <= 0:
        return np.zeros(0, dtype=np.float32)
    model = _resolve_model(seed, sample_rate, model)
    rng = _component_rng(model.seed, 0, _FAN)
    low, high = model.fan_band_hz
    noise = bandpass_noise(rng, size, sample_rate, low, min(high, sample_rate * 0.44))
    noise = one_pole_highpass(noise, sample_rate, low)
    noise = one_pole_lowpass(noise, sample_rate, min(high * 0.85, sample_rate * 0.44))
    if envelope is None:
        time = np.arange(size, dtype=np.float32) / np.float32(sample_rate)
        env = np.asarray(
            np.minimum(1.0, time / 0.09) * np.minimum(1.0, (size / sample_rate - time) / 0.10),
            dtype=np.float32,
        )
    else:
        env = np.asarray(envelope[:size], dtype=np.float32)
    return np.asarray(noise * env * np.float32(amplitude), dtype=np.float32)


def _roller_train(
    sample_rate: int,
    duration_s: float,
    seed: int,
    event_index: int,
    *,
    rate_hz: float = 13.0,
    amplitude: float = 0.10,
) -> FloatAudio:
    size = max(1, round(duration_s * sample_rate))
    result = np.zeros(size, dtype=np.float32)
    rng = _component_rng(seed, event_index, _GEAR)
    cursor = 0.0
    while cursor < size:
        burst_size = min(
            size - round(cursor),
            max(3, round(float(rng.uniform(0.003, 0.010)) * sample_rate)),
        )
        if burst_size <= 0:
            break
        burst = bandpass_noise(
            rng,
            burst_size,
            sample_rate,
            130.0,
            min(2_600.0, sample_rate * 0.44),
        )
        time = np.arange(burst_size, dtype=np.float32) / np.float32(sample_rate)
        burst *= np.exp(-time / np.float32(0.004 + float(rng.uniform(0.0, 0.006))))
        gain = amplitude * float(rng.uniform(0.94, 1.06))
        mix_at(result, np.asarray(burst * np.float32(gain), dtype=np.float32), round(cursor))
        cursor += sample_rate / rate_hz * float(rng.uniform(0.92, 1.08))
    return result


def _paper_rustle(
    sample_rate: int,
    duration_s: float,
    seed: int,
    event_index: int,
    *,
    amplitude: float = 0.17,
) -> FloatAudio:
    size = max(1, round(duration_s * sample_rate))
    rng = _component_rng(seed, event_index, _PAPER)
    paper = bandpass_noise(rng, size, sample_rate, 450.0, min(4_500.0, sample_rate * 0.44))
    detail = bandpass_noise(
        rng,
        size,
        sample_rate,
        min(1_800.0, sample_rate * 0.20),
        min(8_500.0, sample_rate * 0.44),
    )
    low = bandpass_noise(rng, size, sample_rate, 120.0, min(900.0, sample_rate * 0.40))
    times = duration_s
    envelope = envelope_from_points(
        [
            (0.0, 0.08),
            (0.07 * times, 0.95),
            (0.20 * times, 0.68),
            (0.62 * times, 0.55),
            (0.78 * times, 0.96),
            (0.92 * times, 0.36),
            (times, 0.0),
        ],
        sample_rate,
        size,
    )
    result = (paper * 0.72 + detail * 0.30 + low * 0.07) * envelope
    micro_rng = _component_rng(seed, event_index, _VARIATION)
    event_count = max(2, round(duration_s * float(micro_rng.uniform(8.0, 15.0))))
    for _ in range(event_count):
        start = int(micro_rng.integers(0, max(1, size)))
        burst_size = min(
            size - start,
            max(2, round(float(micro_rng.uniform(0.002, 0.012)) * sample_rate)),
        )
        if burst_size <= 0:
            continue
        burst = bandpass_noise(
            micro_rng,
            burst_size,
            sample_rate,
            1_400.0,
            min(8_500.0, sample_rate * 0.44),
        )
        burst *= np.exp(
            -np.arange(burst_size, dtype=np.float32) / np.float32(max(1, 0.004 * sample_rate))
        )
        mix_at(result, burst * np.float32(0.16), start)
    return np.asarray(result * np.float32(amplitude), dtype=np.float32)


def _mixed(size: int, layers: list[tuple[FloatAudio, int]]) -> FloatAudio:
    result = np.zeros(max(1, size), dtype=np.float32)
    for signal, start in layers:
        mix_at(result, signal, start)
    return result


def _printer_beep(*, sample_rate: int, seed: int, event_index: int) -> FloatAudio:
    preset = TonePreset(
        frequency_hz=880.0,
        harmonic_gains=(1.0, 0.30),
        attack_s=0.004,
        release_s=0.035,
        bandwidth_hz=(500.0, 3_700.0),
        transient_click=0.015,
    )
    return render_tone_pattern(
        preset=preset,
        pulses=(TonePulse(start_s=0.0, duration_s=0.09, level=0.14),),
        duration_s=0.12,
        sample_rate=sample_rate,
        rng=component_rng(seed, _PRINTER_SALT, _TONE, event_index),
    )


def _tray_body(model: PrinterModel) -> ModalBody:
    return ModalBody(
        modes=tuple(
            SlidingMode(
                frequency_hz=mode.frequency_hz,
                decay_s=mode.decay_s,
                gain=mode.input_gain * mode.radiation_gain,
            )
            for mode in model.tray_modes.modes
        )
    )


def _tray_sliding(
    motion: MotionCurve,
    *,
    sample_rate: int,
    seed: int,
    event_index: int,
    model: PrinterModel,
    paper_load: str,
) -> FloatAudio:
    velocity = np.abs(motion.velocity)
    peak = float(np.max(velocity)) if velocity.size else 0.0
    activity = np.asarray(velocity / max(peak, 1e-8), dtype=np.float32)
    load = {"empty": 0.0, "partial": 0.5, "full": 1.0}[paper_load]
    profile = FrictionProfile(
        base_gain=0.52 + 0.12 * load,
        roughness=0.64,
        noise_band_hz=(90.0, 3_200.0),
        stick_strength=0.11 + 0.03 * load,
        slip_strength=0.42,
        f0_hz=(180.0, 480.0),
        harmonic_rolloff=(1.6, 2.2),
        chaos_amount=0.25,
    )
    return render_sliding_source(
        motion=motion,
        profile=profile,
        sample_rate=sample_rate,
        seed=seed,
        event_index=event_index,
        body=_tray_body(model),
        activity=activity,
    )


def render_printer_print(*, sample_rate: int, pages: int, speed: str, seed: int) -> FloatAudio:
    page_duration = _PAGE_DURATION[speed]
    startup = 0.12
    spacing = page_duration * 0.82
    total_duration = startup + spacing * (pages - 1) + page_duration + 0.18
    size = max(1, round(total_duration * sample_rate))
    motor_env = envelope_from_points(
        [
            (0.0, 0.0),
            (0.08, 0.65),
            (0.32, 0.46),
            (total_duration - 0.10, 0.34),
            (total_duration, 0.0),
        ],
        sample_rate,
        size,
    )
    layers: list[tuple[FloatAudio, int]] = [
        (_motor_whir(sample_rate, size, seed, 0, amplitude=0.36, envelope=motor_env), 0),
        (_fan_noise(sample_rate, size, seed, 0, amplitude=0.028, envelope=motor_env), 0),
    ]
    for page in range(pages):
        event = page + 1
        start_s = startup + spacing * page
        start = round(start_s * sample_rate)
        click = _click(sample_rate, seed, event, strength=0.82)
        paper_duration = page_duration * 0.79
        rollers = _roller_train(
            sample_rate,
            page_duration * 0.81,
            seed,
            event,
            rate_hz=14.0 if speed != "fast" else 17.0,
            amplitude=0.09,
        )
        paper = _paper_rustle(sample_rate, paper_duration, seed, event, amplitude=0.19)
        flap = _click(sample_rate, seed, event + 100, strength=0.58)
        layers.extend(
            [
                (click, start),
                (rollers, start + round(0.045 * sample_rate)),
                (paper, start + round(0.10 * sample_rate)),
                (flap, start + round(page_duration * 0.84 * sample_rate)),
            ]
        )
    return _mixed(size, layers)


def render_printer_tray_open(
    *,
    sample_rate: int,
    speed: str,
    paper_load: str,
    seed: int,
    model: PrinterModel | None = None,
) -> FloatAudio:
    model = _resolve_model(seed, sample_rate, model)
    duration = _TRAY_DURATION[speed]
    motion = minimum_jerk_motion(duration_s=duration, sample_rate=sample_rate)
    sliding = _tray_sliding(
        motion,
        sample_rate=sample_rate,
        seed=seed,
        event_index=0,
        model=model,
        paper_load=paper_load,
    )
    latch = render_terminal_closure(
        contact=model.latch_contact,
        velocity_m_s=0.14,
        modes=model.tray_modes,
        sample_rate=sample_rate,
        seed=seed,
        event_index=1,
        output_gain=15.0,
        tail_s=0.06,
    )
    stop_velocity = {"slow": 0.24, "normal": 0.34, "fast": 0.48}[speed]
    stop = render_terminal_closure(
        contact=model.tray_stop_contact,
        velocity_m_s=stop_velocity,
        modes=model.tray_modes,
        sample_rate=sample_rate,
        seed=seed,
        event_index=3,
        output_gain=18.0,
        tail_s=0.18,
    )
    size = motion.position.size + round(0.19 * sample_rate)
    layers: list[tuple[FloatAudio, int]] = [
        (latch, 0),
        (sliding, round(0.035 * sample_rate)),
        (stop, motion.position.size),
    ]
    load_gain = {"empty": 0.0, "partial": 0.24, "full": 0.55}[paper_load]
    if load_gain:
        paper = _paper_rustle(sample_rate, 0.20, seed, 4, amplitude=load_gain)
        layers.append((paper, round(duration * 0.43 * sample_rate)))
        if paper_load == "full":
            layers.append(
                (
                    _click(sample_rate, seed, 5, strength=0.16, model=model, body="tray"),
                    round(duration * 0.83 * sample_rate),
                )
            )
    return _mixed(size, layers)


def render_printer_tray_close(
    *,
    sample_rate: int,
    speed: str,
    paper_load: str,
    force: str,
    seed: int,
    model: PrinterModel | None = None,
) -> FloatAudio:
    model = _resolve_model(seed, sample_rate, model)
    duration = _TRAY_DURATION[speed]
    opening_motion = minimum_jerk_motion(duration_s=duration, sample_rate=sample_rate)
    motion = MotionCurve(
        position=np.asarray(1.0 - opening_motion.position, dtype=np.float32),
        velocity=np.asarray(-opening_motion.velocity, dtype=np.float32),
        acceleration=np.asarray(-opening_motion.acceleration, dtype=np.float32),
    )
    sliding = _tray_sliding(
        motion,
        sample_rate=sample_rate,
        seed=seed,
        event_index=6,
        model=model,
        paper_load=paper_load,
    )
    load_scale = {"empty": 0.0, "partial": 0.5, "full": 1.0}[paper_load]
    terminal_velocity = {"gentle": 0.22, "normal": 0.42, "firm": 0.68}[force]
    terminal = render_terminal_closure(
        contact=model.tray_stop_contact,
        velocity_m_s=terminal_velocity * (1.0 + 0.08 * load_scale),
        modes=model.tray_modes,
        sample_rate=sample_rate,
        seed=seed,
        event_index=7,
        output_gain=20.0,
        tail_s=0.18,
    )
    latch = _click(
        sample_rate, seed, 8, strength=0.42, model=model, body="tray"
    )
    size = motion.position.size + round(0.19 * sample_rate)
    layers: list[tuple[FloatAudio, int]] = [
        (sliding, round(0.035 * sample_rate)),
        (terminal, motion.position.size),
        (latch, motion.position.size + round(0.025 * sample_rate)),
    ]
    paper_gain = {"empty": 0.0, "partial": 0.20, "full": 0.46}[paper_load]
    if paper_gain:
        paper = _paper_rustle(sample_rate, 0.22, seed, 9, amplitude=paper_gain)
        layers.append((paper, round(duration * 0.25 * sample_rate)))
    return _mixed(size, layers)


def render_printer_power_switch(*, sample_rate: int, state: str, seed: int) -> FloatAudio:
    if state == "off":
        duration = 0.62
        size = round(duration * sample_rate)
        t = np.arange(size, dtype=np.float32) / np.float32(sample_rate)
        tail_env = np.asarray(np.exp(-t / np.float32(0.23)), dtype=np.float32)
        layers: list[tuple[FloatAudio, int]] = [
            (_click(sample_rate, seed, 0, strength=0.86), 0),
            (_click(sample_rate, seed, 1, strength=0.62), round(0.13 * sample_rate)),
            (_motor_whir(sample_rate, size, seed, 2, amplitude=0.11, envelope=tail_env), 0),
            (_fan_noise(sample_rate, size, seed, 3, amplitude=0.048, envelope=tail_env), 0),
        ]
        return _mixed(size, layers)
    duration = 0.40
    size = round(duration * sample_rate)
    env = envelope_from_points(
        [(0.0, 0.0), (0.12, 0.0), (0.24, 0.65), (0.36, 0.22), (0.40, 0.0)],
        sample_rate,
        size,
    )
    return _mixed(
        size,
        [
            (_click(sample_rate, seed, 0, strength=0.84), 0),
            (_click(sample_rate, seed, 1, strength=0.56), round(0.105 * sample_rate)),
            (_fan_noise(sample_rate, size, seed, 2, amplitude=0.046, envelope=env), 0),
            (_motor_whir(sample_rate, size, seed, 3, amplitude=0.12, envelope=env), 0),
        ],
    )


def render_printer_restart(
    *, sample_rate: int, speed: str, seed: int, beep: bool = False
) -> FloatAudio:
    duration = _RESTART_DURATION[speed]
    size = round(duration * sample_rate)
    bed_env = envelope_from_points(
        [
            (0.0, 0.0),
            (0.14 * duration, 0.68),
            (0.35 * duration, 0.36),
            (0.60 * duration, 0.52),
            (0.82 * duration, 0.24),
            (duration, 0.0),
        ],
        sample_rate,
        size,
    )
    layers: list[tuple[FloatAudio, int]] = [
        (_click(sample_rate, seed, 0, strength=0.72), 0),
        (_fan_noise(sample_rate, size, seed, 0, amplitude=0.042, envelope=bed_env), 0),
        (
            render_electromechanical_hum(
                sample_rate=sample_rate,
                size=size,
                seed=seed,
                event_index=0,
                amplitude=0.012,
                envelope=bed_env,
            ),
            0,
        ),
        (
            _motor_whir(sample_rate, size, seed, 0, amplitude=0.27, envelope=bed_env),
            round(0.12 * sample_rate),
        ),
    ]
    phases = (0.23, 0.43, 0.63)
    for index, fraction in enumerate(phases, start=1):
        duration_s = duration * (0.060 if index != 2 else 0.075)
        start = round(duration * fraction * sample_rate)
        gear = _roller_train(
            sample_rate,
            duration_s,
            seed,
            index,
            rate_hz=18.0 + index,
            amplitude=0.12,
        )
        servo = _motor_whir(
            sample_rate,
            max(1, round(duration_s * sample_rate)),
            seed,
            index,
            amplitude=0.26,
        )
        layers.extend([(gear, start), (servo, start)])
        if index < 3:
            layers.append((_click(sample_rate, seed, index + 10, strength=0.36), start))
    layers.append(
        (_click(sample_rate, seed, 30, strength=0.42), round(duration * 0.88 * sample_rate))
    )
    if beep:
        layers.append(
            (_printer_beep(sample_rate=sample_rate, seed=seed, event_index=31), size - round(0.12 * sample_rate))
        )
    return _mixed(size, layers)


def render_printer_wake(
    *, sample_rate: int, depth: str, seed: int, beep: bool = False
) -> FloatAudio:
    duration = _WAKE_DURATION[depth]
    size = round(duration * sample_rate)
    layers: list[tuple[FloatAudio, int]] = [(_click(sample_rate, seed, 0, strength=0.52), 0)]
    fan_start = round(0.06 * sample_rate)
    fan_size = size - fan_start
    fan_env = envelope_from_points(
        [(0.0, 0.0), (0.10, 0.62), (max(0.11, duration - 0.12), 0.30), (duration, 0.0)],
        sample_rate,
        fan_size,
    )
    layers.append(
        (_fan_noise(sample_rate, fan_size, seed, 0, amplitude=0.034, envelope=fan_env), fan_start)
    )
    layers.append(
        (
            render_electromechanical_hum(
                sample_rate=sample_rate,
                size=fan_size,
                seed=seed,
                event_index=1,
                amplitude=0.008,
                envelope=fan_env,
            ),
            fan_start,
        )
    )
    twitch_duration = 0.22 if depth == "light" else 0.34
    twitch_start = round(0.19 * sample_rate)
    twitch_size = max(1, round(twitch_duration * sample_rate))
    layers.append((_motor_whir(sample_rate, twitch_size, seed, 1, amplitude=0.19), twitch_start))
    layers.append(
        (
            _roller_train(
                sample_rate,
                0.14 if depth == "light" else 0.22,
                seed,
                2,
                rate_hz=12.0,
                amplitude=0.075,
            ),
            round(0.25 * sample_rate),
        )
    )
    settle_time = 0.53 if depth == "light" else 0.86
    layers.append((_click(sample_rate, seed, 3, strength=0.30), round(settle_time * sample_rate)))
    if depth == "deep":
        layers.append(
            (
                _roller_train(sample_rate, 0.11, seed, 4, rate_hz=15.0, amplitude=0.065),
                round(0.67 * sample_rate),
            )
        )
    if beep:
        layers.append(
            (_printer_beep(sample_rate=sample_rate, seed=seed, event_index=32), size - round(0.12 * sample_rate))
        )
    return _mixed(size, layers)

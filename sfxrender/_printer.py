"""Shared deterministic synthesis layers for small electromechanical printers."""

from __future__ import annotations

import math

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

_PRINTER_SALT = 0x50524E54
_SWITCH_CONTACT = 1
_MOTOR = 2
_GEAR = 3
_PAPER = 4
_FAN = 5
_CHASSIS = 6
_VARIATION = 7
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
) -> FloatAudio:
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
    mode_rng = _component_rng(seed, event_index, _CHASSIS)
    modes = tuple(mode for mode in _CHASSIS_MODES if mode[0] < sample_rate * 0.46)
    body = modal_bank(excitation, sample_rate, mode_rng, modes)
    return np.asarray((excitation * 0.58 + body * 0.24) * np.float32(strength), dtype=np.float32)


def _motor_whir(
    sample_rate: int,
    size: int,
    seed: int,
    event_index: int,
    *,
    amplitude: float = 0.22,
    envelope: FloatAudio | None = None,
) -> FloatAudio:
    if size <= 0:
        return np.zeros(0, dtype=np.float32)
    rng = _component_rng(seed, event_index, _MOTOR)
    variation = _component_rng(seed, event_index, _VARIATION)
    time = np.arange(size, dtype=np.float64) / sample_rate
    duration = size / sample_rate
    base_hz = float(rng.uniform(76.0, 112.0))
    attack = max(0.025, min(0.14, duration * 0.24))
    ramp = np.clip(time / attack, 0.0, 1.0)
    release_s = max(0.045, min(0.18, duration * 0.26))
    release = np.clip((duration - time) / release_s, 0.0, 1.0)
    motion = ramp * release
    detune = float(rng.uniform(-1.8, 1.8))
    frequency = base_hz * (0.78 + 0.22 * ramp) + detune
    frequency += 1.1 * np.sin(2.0 * math.pi * float(rng.uniform(1.1, 2.8)) * time)
    phase = np.cumsum(2.0 * math.pi * frequency / sample_rate, dtype=np.float64)
    gear_rate = float(rng.uniform(8.0, 19.0))
    ripple = 0.84 + 0.16 * np.sin(2.0 * math.pi * gear_rate * time + float(rng.uniform(-1, 1)))
    tone = np.zeros(size, dtype=np.float64)
    for harmonic, gain in ((1, 1.0), (2, 0.42), (3, 0.23), (4, 0.12)):
        if base_hz * harmonic < sample_rate * 0.46:
            tone += np.sin(phase * harmonic + float(rng.uniform(-0.12, 0.12))) * gain
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
    signal = (tone * ripple * 0.21 + texture.astype(np.float64) * 0.11) * env
    return np.asarray(signal * amplitude, dtype=np.float32)


def _fan_noise(
    sample_rate: int,
    size: int,
    seed: int,
    event_index: int,
    *,
    amplitude: float = 0.055,
    envelope: FloatAudio | None = None,
) -> FloatAudio:
    if size <= 0:
        return np.zeros(0, dtype=np.float32)
    rng = _component_rng(seed, event_index, _FAN)
    noise = bandpass_noise(rng, size, sample_rate, 75.0, min(4_000.0, sample_rate * 0.44))
    noise = one_pole_highpass(noise, sample_rate, 75.0)
    noise = one_pole_lowpass(noise, sample_rate, min(3_400.0, sample_rate * 0.44))
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
    *, sample_rate: int, speed: str, paper_load: str, seed: int
) -> FloatAudio:
    duration = _TRAY_DURATION[speed]
    total_duration = duration + 0.19
    size = max(1, round(total_duration * sample_rate))
    motion = minimum_jerk_motion(duration_s=duration, sample_rate=sample_rate)
    velocity = np.abs(motion.velocity)
    velocity /= np.float32(max(float(np.max(velocity)), 1e-8))
    rng = _component_rng(seed, 0, _GEAR)
    rail_low = bandpass_noise(
        rng, motion.position.size, sample_rate, 90.0, min(900.0, sample_rate * 0.40)
    )
    rail_high = bandpass_noise(
        rng, motion.position.size, sample_rate, 600.0, min(3_500.0, sample_rate * 0.44)
    )
    rail = np.asarray(
        (rail_low * 0.58 + rail_high * 0.44) * (0.10 + 0.90 * velocity) * 0.28,
        dtype=np.float32,
    )
    stick = _roller_train(sample_rate, duration, seed, 2, rate_hz=6.0, amplitude=0.045)
    latch = _click(sample_rate, seed, 1, strength=0.62)
    stop = _click(sample_rate, seed, 3, strength=0.50)
    layers: list[tuple[FloatAudio, int]] = [
        (latch, 0),
        (rail, round(0.035 * sample_rate)),
        (stick, round(0.06 * sample_rate)),
        (stop, round(duration * sample_rate)),
    ]
    load_gain = {"empty": 0.0, "partial": 0.24, "full": 0.55}[paper_load]
    if load_gain:
        paper = _paper_rustle(sample_rate, 0.20, seed, 4, amplitude=load_gain)
        paper = np.asarray(0.42 * np.tanh(paper / np.float32(0.42)), dtype=np.float32)
        layers.append((paper, round(duration * 0.43 * sample_rate)))
        if paper_load == "full":
            layers.append(
                (_click(sample_rate, seed, 5, strength=0.16), round(duration * 0.83 * sample_rate))
            )
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


def render_printer_restart(*, sample_rate: int, speed: str, seed: int) -> FloatAudio:
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
    return _mixed(size, layers)


def render_printer_wake(*, sample_rate: int, depth: str, seed: int) -> FloatAudio:
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
    return _mixed(size, layers)

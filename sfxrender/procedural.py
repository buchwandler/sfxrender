"""Dependency-light procedural MVP effects."""

from __future__ import annotations

import math

import numpy as np

from ._params import choice, integer, number
from .types import FloatAudio, RenderContext, RenderedSound, SfxSpec


def _rng(spec: SfxSpec, event_index: int = 0) -> np.random.Generator:
    """Return a stable per-event generator; omitted seeds use a reproducible default."""
    seed = spec.seed if spec.seed is not None else 0
    return np.random.default_rng(np.random.SeedSequence([seed, event_index]))


def _fade_out(signal: FloatAudio, sample_rate: int, seconds: float) -> FloatAudio:
    n = min(signal.size, max(1, int(seconds * sample_rate)))
    out = signal.copy()
    out[-n:] *= np.linspace(1.0, 0.0, n, dtype=np.float32)
    return out


def _normalize_template(signal: FloatAudio, peak: float = 0.8) -> FloatAudio:
    """Condition a synthesis template before applying semantic gain."""
    maximum = float(np.max(np.abs(signal))) if signal.size else 0.0
    if maximum > 0.0:
        signal = signal * np.float32(peak / maximum)
    return np.asarray(signal, dtype=np.float32)


def _limit_peak(signal: FloatAudio, peak: float = 0.95) -> FloatAudio:
    """Attenuate unsafe output without boosting quiet renders."""
    maximum = float(np.max(np.abs(signal))) if signal.size else 0.0
    if maximum > peak:
        signal = signal * np.float32(peak / maximum)
    return np.asarray(signal, dtype=np.float32)


def _smooth(signal: FloatAudio, taps: int) -> FloatAudio:
    """Apply a short moving-average low-pass without changing the signal length."""
    if signal.size == 0:
        return signal.copy()
    taps = max(1, min(int(taps), signal.size))
    if taps == 1:
        return signal.copy()
    kernel = np.full(taps, 1.0 / taps, dtype=np.float32)
    full = np.convolve(signal, kernel, mode="full")
    start = (taps - 1) // 2
    return full[start : start + signal.size].astype(np.float32)


def _mix_at(destination: FloatAudio, source: FloatAudio, start: int) -> None:
    """Add a source event into a destination, clipping safely at either end."""
    if start < 0:
        source = source[-start:]
        start = 0
    if start >= destination.size or source.size == 0:
        return
    end = min(destination.size, start + source.size)
    destination[start:end] += source[: end - start]


def _unit_rms(signal: FloatAudio) -> FloatAudio:
    rms = float(np.sqrt(np.mean(np.square(signal, dtype=np.float64)))) if signal.size else 0.0
    if rms <= 1e-12:
        return signal.copy()
    return np.asarray(signal / np.float32(rms), dtype=np.float32)


def _colored_burst(
    rng: np.random.Generator,
    sample_rate: int,
    size: int,
    *,
    decay: float,
    cutoff_hz: float,
    brightness: float,
    attack: float = 0.0006,
) -> FloatAudio:
    """Make a deterministic noise burst with a controllable soft/bright balance."""
    if size <= 0:
        return np.zeros(0, dtype=np.float32)
    white = rng.normal(0.0, 1.0, size).astype(np.float32)
    low_taps = max(2, int(sample_rate / max(2.0 * cutoff_hz, 1.0)))
    low = _unit_rms(_smooth(white, low_taps))
    high_taps = max(2, int(sample_rate / 4500.0))
    high = _unit_rms(white - _smooth(white, high_taps))
    mix = float(np.clip(brightness, 0.0, 1.0))
    colored = low * np.float32(1.0 - mix) + high * np.float32(mix)
    t = np.arange(size, dtype=np.float32) / np.float32(sample_rate)
    envelope = np.exp(-t / np.float32(max(decay, 1.0 / sample_rate)))
    attack_env = 1.0 - np.exp(-t / np.float32(max(attack, 1.0 / sample_rate)))
    return np.asarray(colored * envelope * attack_env, dtype=np.float32)


def _modal_bank(
    t: FloatAudio,
    rng: np.random.Generator,
    modes: tuple[tuple[float, float, float], ...],
    *,
    frequency_jitter: float,
    decay_jitter: float,
) -> FloatAudio:
    """Sum independently varied, deliberately inharmonic damped body modes."""
    body = np.zeros(t.size, dtype=np.float32)
    if t.size == 0:
        return body
    for frequency, decay, gain in modes:
        varied_frequency = frequency * float(
            rng.uniform(1.0 - frequency_jitter, 1.0 + frequency_jitter)
        )
        varied_decay = decay * float(rng.uniform(1.0 - decay_jitter, 1.0 + decay_jitter))
        phase = float(rng.uniform(-math.pi, math.pi))
        envelope = np.exp(-t / np.float32(max(varied_decay, 1.0 / 48_000)))
        body += (
            np.float32(gain)
            * envelope
            * np.sin(np.float32(2.0 * math.pi * varied_frequency) * t + np.float32(phase))
        )
    return body


def _event_starts(
    *,
    count: int,
    interval: float,
    sample_rate: int,
    spec: SfxSpec,
    jitter_fraction: float,
) -> list[int]:
    """Return seeded, bounded event starts around a nominal interval."""
    if count <= 0:
        return []
    rng = _rng(spec, 91_337)
    starts = [0]
    for _ in range(1, count):
        variation = float(rng.normal(0.0, max(0.0, jitter_fraction)))
        gap = interval * float(np.clip(1.0 + variation, 0.86, 1.14))
        starts.append(starts[-1] + max(1, round(gap * sample_rate)))
    return starts


def _chirped_body(
    sample_rate: int,
    size: int,
    rng: np.random.Generator,
    *,
    start_frequency: float,
    end_frequency: float,
    decay: float,
    gain: float,
) -> FloatAudio:
    t = np.arange(size, dtype=np.float32) / np.float32(sample_rate)
    duration = max(float(t[-1]) if t.size else 0.0, 1.0 / sample_rate)
    phase = (
        2.0
        * math.pi
        * (start_frequency * t + 0.5 * (end_frequency - start_frequency) / duration * t * t)
    )
    phase_offset = np.float32(rng.uniform(-math.pi, math.pi))
    envelope = np.exp(-t / np.float32(max(decay, 1.0 / sample_rate)))
    return np.asarray(np.sin(phase + phase_offset) * envelope * np.float32(gain), dtype=np.float32)


def _knock_hit(
    *,
    sample_rate: int,
    rng: np.random.Generator,
    modes: tuple[tuple[float, float, float], ...],
    body_profile: tuple[float, float, float, float],
    contact_brightness: float,
    contact_gain: float,
    diffusion_brightness: float,
    diffusion_gain: float,
) -> FloatAudio:
    longest_decay = max(decay for _, decay, _ in modes)
    duration = max(0.24, longest_decay * 5.0 + 0.055)
    size = max(1, round(duration * sample_rate))
    t = np.asarray(np.arange(size, dtype=np.float32) / np.float32(sample_rate), dtype=np.float32)
    start_frequency, end_frequency, body_decay, body_gain = body_profile
    frequency_scale = float(rng.uniform(0.975, 1.025))
    body = _chirped_body(
        sample_rate,
        size,
        rng,
        start_frequency=start_frequency * frequency_scale,
        end_frequency=end_frequency * frequency_scale,
        decay=body_decay * float(rng.uniform(0.9, 1.1)),
        gain=body_gain,
    )
    modes_layer = _modal_bank(
        t,
        rng,
        modes,
        frequency_jitter=0.035,
        decay_jitter=0.11,
    )
    hit = body + modes_layer * np.float32(0.62)
    hit += _colored_burst(
        rng,
        sample_rate,
        size,
        decay=float(rng.uniform(0.0025, 0.0045)),
        cutoff_hz=1800.0,
        brightness=contact_brightness,
        attack=0.00025,
    ) * np.float32(contact_gain * rng.uniform(0.88, 1.12))
    hit += _colored_burst(
        rng,
        sample_rate,
        size,
        decay=float(rng.uniform(0.014, 0.022)),
        cutoff_hz=1250.0,
        brightness=max(0.08, contact_brightness * 0.55),
        attack=0.0015,
    ) * np.float32(contact_gain * 0.45)
    diffusion = _colored_burst(
        rng,
        sample_rate,
        size,
        decay=0.075 if longest_decay < 0.12 else 0.11,
        cutoff_hz=1700.0 if diffusion_brightness < 0.5 else 3600.0,
        brightness=diffusion_brightness,
        attack=0.001,
    )
    slow_taps = max(2, int(sample_rate * 0.035))
    slow = _unit_rms(_smooth(rng.normal(0.0, 1.0, size).astype(np.float32), slow_taps))
    hit += diffusion * np.float32(diffusion_gain) * (1.0 + np.float32(0.035) * slow)
    if rng.random() < 0.24:
        tap_size = max(1, int(0.014 * sample_rate))
        tap = _colored_burst(
            rng,
            sample_rate,
            tap_size,
            decay=0.004,
            cutoff_hz=2400.0,
            brightness=min(0.9, contact_brightness + 0.15),
            attack=0.0003,
        ) * np.float32(contact_gain * rng.uniform(0.12, 0.22))
        _mix_at(hit, tap, int(rng.uniform(0.012, 0.032) * sample_rate))
    return _normalize_template(hit)


def knock(spec: SfxSpec, context: RenderContext) -> RenderedSound:
    params = spec.parameters
    material = choice(params, "material", "wood", {"wood", "oak", "metal", "wall"})
    count = integer(params, "count", 1, minimum=1, maximum=16)
    force = number(params, "force", 0.65, minimum=0.05, maximum=1.0)
    interval = number(params, "interval", 0.22, minimum=0.08, maximum=2.0)
    modes_by_material: dict[str, tuple[tuple[float, float, float], ...]] = {
        "wood": (
            (115.0, 0.070, 0.28),
            (185.0, 0.090, 0.30),
            (315.0, 0.065, 0.25),
            (515.0, 0.050, 0.20),
            (820.0, 0.030, 0.15),
            (1250.0, 0.018, 0.09),
        ),
        "oak": (
            (90.0, 0.095, 0.32),
            (150.0, 0.115, 0.30),
            (255.0, 0.085, 0.25),
            (410.0, 0.060, 0.19),
            (680.0, 0.038, 0.14),
            (1080.0, 0.022, 0.08),
        ),
        "wall": (
            (75.0, 0.035, 0.26),
            (125.0, 0.045, 0.22),
            (230.0, 0.030, 0.18),
            (390.0, 0.022, 0.15),
            (650.0, 0.015, 0.10),
        ),
        "metal": (
            (260.0, 0.100, 0.20),
            (430.0, 0.145, 0.19),
            (710.0, 0.160, 0.17),
            (1180.0, 0.130, 0.15),
            (1900.0, 0.095, 0.13),
            (3000.0, 0.055, 0.10),
        ),
    }
    body_by_material: dict[str, tuple[float, float, float, float]] = {
        "wood": (135.0, 78.0, 0.075, 0.34),
        "oak": (110.0, 72.0, 0.11, 0.40),
        "wall": (90.0, 60.0, 0.038, 0.30),
        "metal": (185.0, 125.0, 0.065, 0.18),
    }
    contact_brightness = {"wood": 0.42, "oak": 0.36, "wall": 0.58, "metal": 0.82}[material]
    contact_gain = {"wood": 0.18, "oak": 0.17, "wall": 0.16, "metal": 0.22}[material]
    diffusion_brightness = {"wood": 0.36, "oak": 0.30, "wall": 0.22, "metal": 0.70}[material]
    diffusion_gain = {"wood": 0.11, "oak": 0.12, "wall": 0.15, "metal": 0.08}[material]
    modes = modes_by_material[material]
    starts = _event_starts(
        count=count,
        interval=interval,
        sample_rate=context.sample_rate,
        spec=spec,
        jitter_fraction=0.008,
    )
    longest_decay = max(decay for _, decay, _ in modes)
    hit_size = max(1, round(max(0.24, longest_decay * 5.0 + 0.055) * context.sample_rate))
    total = starts[-1] + hit_size
    result = np.zeros(total, dtype=np.float32)
    for index, start in enumerate(starts):
        rng = _rng(spec, index)
        hit = _knock_hit(
            sample_rate=context.sample_rate,
            rng=rng,
            modes=modes,
            body_profile=body_by_material[material],
            contact_brightness=contact_brightness,
            contact_gain=contact_gain,
            diffusion_brightness=diffusion_brightness,
            diffusion_gain=diffusion_gain,
        )
        level = force * float(rng.uniform(0.94, 1.04))
        _mix_at(result, hit * np.float32(level), start)
    return RenderedSound(_limit_peak(result), context.sample_rate, spec)


def _footstep_hit(
    *,
    sample_rate: int,
    surface: str,
    footwear: str,
    rng: np.random.Generator,
    force: float,
    side: float,
) -> FloatAudio:
    """Synthesize one short heel-to-sole event with surface and footwear mechanics."""
    surface_modes: dict[str, tuple[tuple[float, float, float], ...]] = {
        "wood": (
            (145.0, 0.100, 0.38),
            (275.0, 0.078, 0.32),
            (520.0, 0.052, 0.24),
            (830.0, 0.035, 0.15),
            (1180.0, 0.024, 0.08),
        ),
        "stone": (
            (180.0, 0.055, 0.28),
            (370.0, 0.045, 0.23),
            (690.0, 0.033, 0.20),
            (1280.0, 0.023, 0.16),
            (1850.0, 0.015, 0.10),
        ),
        "carpet": (
            (90.0, 0.045, 0.30),
            (175.0, 0.032, 0.23),
            (320.0, 0.023, 0.14),
        ),
        "gravel": (
            (125.0, 0.038, 0.20),
            (250.0, 0.030, 0.14),
            (480.0, 0.022, 0.10),
        ),
    }
    surface_profiles = {
        "wood": (0.34, 0.13, 0.54, 0.36, 0.070),
        "stone": (0.58, 0.11, 0.42, 0.47, 0.046),
        "carpet": (0.08, 0.19, 0.34, 0.16, 0.090),
        "gravel": (0.42, 0.09, 0.28, 0.38, 0.052),
    }
    footwear_profiles = {
        "barefoot": (90.0, 63.0, 0.32, 0.095, 0.42, 0.42, -0.18, 0.292, 0.070),
        "shoes": (122.0, 79.0, 0.35, 0.070, 0.64, 0.55, 0.02, 0.265, 0.052),
        "boots": (108.0, 68.0, 0.44, 0.100, 0.74, 0.70, -0.04, 0.305, 0.058),
        "heels": (185.0, 125.0, 0.17, 0.040, 0.95, 0.90, 0.35, 0.245, 0.064),
    }
    (surface_brightness, texture_gain, modal_gain, surface_contact, surface_decay) = (
        surface_profiles[surface]
    )
    (
        start_frequency,
        end_frequency,
        body_gain,
        body_decay,
        heel_gain,
        sole_gain,
        footwear_brightness,
        duration,
        sole_delay,
    ) = footwear_profiles[footwear]
    size = max(1, round(duration * sample_rate))
    t = np.asarray(np.arange(size, dtype=np.float32) / np.float32(sample_rate), dtype=np.float32)
    force_level = float(np.clip((force - 0.05) / 0.95, 0.0, 1.0))
    force_brightness = float(np.clip((force_level - 0.45) * 0.38, -0.18, 0.20))
    brightness = float(
        np.clip(surface_brightness + footwear_brightness + force_brightness, 0.03, 0.92)
    )
    body_scale = float(rng.uniform(0.98, 1.02)) * (1.0 + 0.012 * side)
    body = _chirped_body(
        sample_rate,
        size,
        rng,
        start_frequency=start_frequency * body_scale,
        end_frequency=end_frequency * body_scale,
        decay=body_decay * float(rng.uniform(0.92, 1.12)),
        gain=body_gain * (0.80 + 0.38 * force_level) * (1.0 + 0.025 * side),
    )
    modes = _modal_bank(
        t,
        rng,
        surface_modes[surface],
        frequency_jitter=0.025,
        decay_jitter=0.12,
    )
    hit = body + modes * np.float32(modal_gain * (0.82 + 0.30 * force_level))
    heel = _colored_burst(
        rng,
        sample_rate,
        size,
        decay=0.0030 if footwear == "heels" else 0.0045,
        cutoff_hz=1600.0 if surface == "carpet" else 2600.0,
        brightness=brightness,
        attack=0.00022,
    )
    hit += heel * np.float32(heel_gain * surface_contact * (0.68 + 0.65 * force_level))
    surface_texture = _colored_burst(
        rng,
        sample_rate,
        size,
        decay=max(surface_decay, body_decay),
        cutoff_hz=950.0 if surface in {"wood", "carpet"} else 1700.0,
        brightness=min(0.72, brightness * 0.72),
        attack=0.003,
    )
    hit += surface_texture * np.float32(texture_gain)
    side_delay = int(side * 0.0015 * sample_rate)
    delayed_start = max(
        1, int((sole_delay + side_delay + rng.uniform(-0.004, 0.004)) * sample_rate)
    )
    sole_size = max(1, size - delayed_start)
    sole = _colored_burst(
        rng,
        sample_rate,
        sole_size,
        decay=0.018 if footwear == "heels" else 0.030,
        cutoff_hz=2100.0 if footwear in {"shoes", "heels"} else 1250.0,
        brightness=float(np.clip(brightness + (0.12 if footwear == "heels" else 0.0), 0.03, 0.95)),
        attack=0.001,
    ) * np.float32(sole_gain * (0.78 + 0.38 * force_level))
    sole_body = _chirped_body(
        sample_rate,
        sole_size,
        rng,
        start_frequency=start_frequency * (1.55 if footwear == "heels" else 1.22),
        end_frequency=end_frequency * 1.18,
        decay=0.028 if surface != "wood" else 0.042,
        gain=0.16 if footwear != "heels" else 0.10,
    )
    _mix_at(hit, sole + sole_body, delayed_start)
    if surface == "gravel":
        grain_count = int(rng.integers(6, 16))
        for _ in range(grain_count):
            grain_size = max(2, int(rng.uniform(0.003, 0.013) * sample_rate))
            grain = _colored_burst(
                rng,
                sample_rate,
                grain_size,
                decay=float(rng.uniform(0.0025, 0.008)),
                cutoff_hz=float(rng.uniform(1200.0, 3000.0)),
                brightness=float(rng.uniform(0.28, 0.72)),
                attack=0.0002,
            )
            grain *= np.float32(rng.uniform(0.06, 0.19) * (0.65 + 0.55 * force_level))
            grain_start = int(rng.uniform(0.012, min(0.17, duration - 0.02)) * sample_rate)
            _mix_at(hit, grain, grain_start)
    elif rng.random() < 0.42 and footwear != "heels":
        release_start = int(rng.uniform(0.105, min(0.17, duration - 0.03)) * sample_rate)
        release_size = max(1, size - release_start)
        release = _colored_burst(
            rng,
            sample_rate,
            release_size,
            decay=0.035,
            cutoff_hz=1000.0 if footwear == "barefoot" else 1800.0,
            brightness=float(np.clip(brightness * 0.70, 0.05, 0.7)),
            attack=0.004,
        ) * np.float32(0.08 if footwear == "barefoot" else 0.06)
        _mix_at(hit, release, release_start)
    return _normalize_template(hit, peak=0.72)


def footsteps(spec: SfxSpec, context: RenderContext) -> RenderedSound:
    params = spec.parameters
    surface = choice(params, "surface", "wood", {"wood", "stone", "gravel", "carpet"})
    footwear = choice(params, "footwear", "shoes", {"barefoot", "shoes", "boots", "heels"})
    count = integer(params, "count", 4, minimum=1, maximum=64)
    force = number(params, "force", 0.6, minimum=0.05, maximum=1.0)
    interval = number(params, "interval", 0.52, minimum=0.18, maximum=2.0)
    starts = _event_starts(
        count=count,
        interval=interval,
        sample_rate=context.sample_rate,
        spec=spec,
        jitter_fraction=0.04,
    )
    duration = {"barefoot": 0.292, "shoes": 0.265, "boots": 0.305, "heels": 0.245}[footwear]
    hit_size = max(1, round(duration * context.sample_rate))
    result = np.zeros(starts[-1] + hit_size, dtype=np.float32)
    for index, start in enumerate(starts):
        rng = _rng(spec, index)
        side = -1.0 if index % 2 else 1.0
        hit = _footstep_hit(
            sample_rate=context.sample_rate,
            surface=surface,
            footwear=footwear,
            rng=rng,
            force=force,
            side=side,
        )
        local_force = force * float(rng.uniform(0.95, 1.05)) * (1.0 + 0.025 * side)
        _mix_at(result, hit * np.float32(local_force), start)
    result = _fade_out(result, context.sample_rate, 0.025)
    return RenderedSound(_limit_peak(result), context.sample_rate, spec)


def phone_ring(spec: SfxSpec, context: RenderContext) -> RenderedSound:
    params = spec.parameters
    style = choice(params, "style", "classic", {"classic", "electronic"})
    count = integer(params, "count", 1, minimum=1, maximum=12)
    interval = number(params, "interval", 1.15, minimum=0.25, maximum=5.0)
    ring_duration = 0.62 if style == "classic" else 0.42
    n = int(ring_duration * context.sample_rate)
    t = np.arange(n, dtype=np.float32) / np.float32(context.sample_rate)
    spacing = int(interval * context.sample_rate)
    total = spacing * (count - 1) + n
    result = np.zeros(total, dtype=np.float32)

    for index in range(count):
        rng = _rng(spec, index)
        level = float(rng.uniform(0.9, 1.0))
        phase = float(rng.uniform(-0.15, 0.15))
        drift = float(rng.uniform(-0.5, 0.5))
        if style == "classic":
            frequency_a = float(rng.uniform(436.0, 444.0))
            frequency_b = float(rng.uniform(476.0, 484.0))
            carrier = 0.62 * np.sin(2.0 * math.pi * (frequency_a * t + 0.5 * drift * t * t) + phase)
            carrier += 0.38 * np.sin(
                2.0 * math.pi * (frequency_b * t + 0.5 * drift * t * t) - phase
            )
            wobble_rate = float(rng.uniform(19.0, 21.0))
            modulation = 0.78 + 0.22 * np.sin(2.0 * math.pi * wobble_rate * t + phase)
            ring = np.asarray(carrier * modulation, dtype=np.float32)
        else:
            frequency_a = float(rng.uniform(868.0, 892.0))
            frequency_b = float(rng.uniform(1300.0, 1340.0))
            carrier = np.sin(2.0 * math.pi * frequency_a * t + phase)
            carrier += 0.45 * np.sin(2.0 * math.pi * frequency_b * t - phase)
            pulse_rate = float(rng.uniform(6.6, 7.4))
            duty = float(rng.uniform(-0.19, -0.11))
            gate = (np.sin(2.0 * math.pi * pulse_rate * t + phase) > duty).astype(np.float32)
            ring = np.asarray(carrier * gate, dtype=np.float32)

        attack = min(n, max(1, int(0.012 * context.sample_rate)))
        release = min(n, max(1, int(0.035 * context.sample_rate)))
        envelope = np.ones(n, dtype=np.float32)
        envelope[:attack] = np.linspace(0.0, 1.0, attack, dtype=np.float32)
        envelope[-release:] *= np.linspace(1.0, 0.0, release, dtype=np.float32)
        ring = np.asarray(ring, dtype=np.float32) * envelope * np.float32(0.62 * level)
        start = index * spacing
        result[start : start + n] += ring
    return RenderedSound(_limit_peak(result), context.sample_rate, spec)


def door_open(spec: SfxSpec, context: RenderContext) -> RenderedSound:
    params = spec.parameters
    material = choice(params, "material", "wood", {"wood", "metal"})
    speed = choice(params, "speed", "normal", {"slow", "normal", "fast"})
    creak = number(params, "creak", 0.65, minimum=0.0, maximum=1.0)
    duration = {"slow": 2.4, "normal": 1.5, "fast": 0.85}[speed]
    rng = _rng(spec)
    n = int(duration * context.sample_rate)
    t = np.arange(n, dtype=np.float32) / np.float32(context.sample_rate)
    progress = np.clip(t / np.float32(duration), 0.0, 1.0)
    easing = {"slow": 1.25, "normal": 1.0, "fast": 0.78}[speed]
    motion = progress**easing
    start_frequency = 155.0 if material == "wood" else 260.0
    end_frequency = 380.0 if material == "wood" else 720.0
    phase = (
        2.0
        * math.pi
        * (start_frequency * t + 0.5 * (end_frequency - start_frequency) / duration * t * t)
    )
    friction = 0.78 + 0.22 * np.sin(2.0 * math.pi * (3.1 if speed == "slow" else 4.8) * t)
    motion_envelope = np.sin(np.pi * motion) ** 1.25
    hinge_tone = np.sin(phase).astype(np.float32) * motion_envelope.astype(np.float32)
    hinge_tone *= friction.astype(np.float32) * np.float32(0.48 * creak)

    noise = rng.normal(0.0, 1.0, n).astype(np.float32)
    texture_level = 0.07 if material == "wood" else 0.1
    texture = noise * motion_envelope.astype(np.float32) * np.float32(texture_level)
    hinge_friction = noise * motion_envelope.astype(np.float32) * np.float32(0.035 * creak)

    # A latch/clunk at the start and a smaller movement stop near the end.
    handle = np.zeros(n, dtype=np.float32)
    handle_n = min(n, max(1, int(0.09 * context.sample_rate)))
    handle_noise = rng.normal(0.0, 1.0, handle_n).astype(np.float32)
    handle_env = np.exp(
        -np.arange(handle_n, dtype=np.float32) / np.float32(0.012 * context.sample_rate)
    )
    handle[:handle_n] += handle_noise * handle_env * np.float32(0.15)
    handle[:handle_n] += (
        np.sin(
            np.float32(2.0 * math.pi * (170.0 if material == "wood" else 290.0))
            * np.arange(handle_n, dtype=np.float32)
            / np.float32(context.sample_rate)
        )
        * handle_env
        * np.float32(0.2)
    )

    stop_n = min(n, max(1, int(0.055 * context.sample_rate)))
    stop_start = max(0, n - stop_n - int(0.04 * context.sample_rate))
    stop_noise = rng.normal(0.0, 1.0, stop_n).astype(np.float32)
    stop_env = np.exp(
        -np.arange(stop_n, dtype=np.float32) / np.float32(0.009 * context.sample_rate)
    )
    handle[stop_start : stop_start + stop_n] += stop_noise * stop_env * np.float32(0.1)

    result = hinge_tone + texture + hinge_friction + handle
    result = _fade_out(result.astype(np.float32), context.sample_rate, 0.05)
    return RenderedSound(_limit_peak(result), context.sample_rate, spec)

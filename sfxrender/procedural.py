"""Dependency-light procedural MVP effects."""

from __future__ import annotations

import math

import numpy as np

from ._dsp import (
    asymmetric_pulse,
    bandpass_noise,
    impact_excitation,
    modal_bank,
)
from ._footsteps import aggregate_footstep, solid_footstep
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


def _limit_peak(signal: FloatAudio, peak: float = 0.95) -> FloatAudio:
    """Attenuate unsafe output without boosting quiet renders."""
    maximum = float(np.max(np.abs(signal))) if signal.size else 0.0
    if maximum > peak:
        signal = signal * np.float32(peak / maximum)
    return np.asarray(signal, dtype=np.float32)


def _mix_at(destination: FloatAudio, source: FloatAudio, start: int) -> None:
    """Add a source event into a destination, clipping safely at either end."""
    if start < 0:
        source = source[-start:]
        start = 0
    if start >= destination.size or source.size == 0:
        return
    end = min(destination.size, start + source.size)
    destination[start:end] += source[: end - start]


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


def _knock_hit(
    *,
    sample_rate: int,
    rng: np.random.Generator,
    modes: tuple[tuple[float, float, float], ...],
    contact_brightness: float,
    contact_gain: float,
    diffusion_brightness: float,
    diffusion_gain: float,
) -> FloatAudio:
    """Drive a material-specific modal response with a short broadband impact."""
    longest_decay = max(decay for _, decay, _ in modes)
    duration = max(0.24, longest_decay * 5.0 + 0.055)
    size = max(1, round(duration * sample_rate))
    time = np.arange(size, dtype=np.float32) / np.float32(sample_rate)
    force_envelope = asymmetric_pulse(
        size,
        sample_rate,
        attack_s=float(rng.uniform(0.00022, 0.00055)),
        decay_s=float(rng.uniform(0.004, 0.010)),
    )
    envelope_peak = float(np.max(force_envelope)) if force_envelope.size else 0.0
    if envelope_peak > 0.0:
        force_envelope /= np.float32(envelope_peak)
    excitation = impact_excitation(
        sample_rate=sample_rate,
        duration=size,
        force_envelope=force_envelope,
        hardness=contact_brightness,
        brightness=contact_brightness,
        rng=rng,
    )
    hit = excitation * np.float32(contact_gain * rng.uniform(0.88, 1.12))
    modes_layer = modal_bank(
        excitation,
        sample_rate,
        rng,
        modes,
        frequency_jitter=0.025,
        decay_jitter=0.10,
    )
    hit += modes_layer * np.float32(0.58)
    diffusion_cutoff = 1_600.0 if diffusion_brightness < 0.5 else 3_800.0
    diffusion_noise = bandpass_noise(
        rng,
        size,
        sample_rate,
        max(180.0, diffusion_cutoff * 0.32),
        min(diffusion_cutoff * 2.1, sample_rate * 0.47),
    )
    diffusion_decay = 0.075 if longest_decay < 0.12 else 0.11
    diffusion_envelope = np.exp(-time / np.float32(diffusion_decay))
    diffusion_envelope *= 1.0 - np.exp(-time / np.float32(0.0012))
    slow_modulation = 0.96 + 0.04 * np.sin(
        np.float32(2.0 * math.pi * float(rng.uniform(17.0, 29.0))) * time
        + np.float32(rng.uniform(-math.pi, math.pi))
    )
    hit += (
        diffusion_noise
        * diffusion_envelope
        * slow_modulation.astype(np.float32)
        * np.float32(diffusion_gain)
    )
    if rng.random() < 0.24:
        tap_size = max(1, round(0.014 * sample_rate))
        tap_force = asymmetric_pulse(
            tap_size,
            sample_rate,
            attack_s=0.0003,
            decay_s=0.004,
            amplitude=float(rng.uniform(0.12, 0.22)),
        )
        tap = impact_excitation(
            sample_rate=sample_rate,
            duration=tap_size,
            force_envelope=tap_force,
            hardness=contact_brightness,
            brightness=min(0.95, contact_brightness + 0.15),
            rng=rng,
        ) * np.float32(contact_gain)
        _mix_at(hit, tap, round(float(rng.uniform(0.012, 0.032)) * sample_rate))
    return np.asarray(hit, dtype=np.float32)


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
    contact_brightness = {"wood": 0.42, "oak": 0.36, "wall": 0.58, "metal": 0.82}[material]
    contact_gain = {"wood": 0.18, "oak": 0.17, "wall": 0.14, "metal": 0.27}[material]
    diffusion_brightness = {"wood": 0.36, "oak": 0.30, "wall": 0.22, "metal": 0.70}[material]
    diffusion_gain = {"wood": 0.11, "oak": 0.12, "wall": 0.11, "metal": 0.16}[material]
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
    """Route one event through its separate solid or aggregate surface model."""
    local_force = force * (1.0 + 0.03 * side)
    if surface == "gravel":
        return aggregate_footstep(
            sample_rate=sample_rate,
            surface=surface,
            footwear=footwear,
            force=local_force,
            rng=rng,
        )
    return solid_footstep(
        sample_rate=sample_rate,
        surface=surface,
        footwear=footwear,
        force=local_force,
        rng=rng,
    )


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
    if surface == "gravel":
        duration += 0.02
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
        local_force = float(rng.uniform(0.95, 1.05)) * (1.0 + 0.025 * side)
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

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


def _resonant_impact(
    *, sample_rate: int, rng: np.random.Generator, modes: tuple[tuple[float, float, float], ...]
) -> FloatAudio:
    """Synthesize one impact from varied body modes and a separate contact transient."""
    duration = max(decay for _, decay, _ in modes) * 5.0
    n = max(1, int(duration * sample_rate))
    t = np.arange(n, dtype=np.float32) / np.float32(sample_rate)
    body = np.zeros(n, dtype=np.float32)
    phase = rng.uniform(-0.22, 0.22, size=len(modes))
    frequency_scale = float(rng.uniform(0.975, 1.025))
    for index, (frequency, decay, gain) in enumerate(modes):
        varied_frequency = frequency * frequency_scale * float(rng.uniform(0.992, 1.008))
        varied_decay = decay * float(rng.uniform(0.9, 1.1))
        envelope = np.exp(-t / np.float32(varied_decay))
        body += (
            np.float32(gain)
            * envelope
            * np.sin(np.float32(2.0 * math.pi * varied_frequency) * t + np.float32(phase[index]))
        )

    noise = rng.normal(0.0, 1.0, n).astype(np.float32)
    high_component = noise - np.roll(noise, 1)
    high_component[0] = noise[0]
    contact_env = np.exp(-t / np.float32(0.0025))
    surface_env = np.exp(-t / np.float32(0.018))
    hit = body * np.float32(0.82)
    hit += noise * contact_env * np.float32(rng.uniform(0.12, 0.2))
    hit += high_component * surface_env * np.float32(0.025)
    return _normalize_template(hit)


def knock(spec: SfxSpec, context: RenderContext) -> RenderedSound:
    params = spec.parameters
    material = choice(params, "material", "wood", {"wood", "oak", "metal", "wall"})
    count = integer(params, "count", 1, minimum=1, maximum=16)
    force = number(params, "force", 0.65, minimum=0.05, maximum=1.0)
    interval = number(params, "interval", 0.22, minimum=0.08, maximum=2.0)
    modes_by_material = {
        "wood": ((190.0, 0.065, 1.0), (430.0, 0.045, 0.55), (980.0, 0.025, 0.22)),
        "oak": ((155.0, 0.085, 1.0), (360.0, 0.055, 0.52), (760.0, 0.032, 0.20)),
        "metal": ((420.0, 0.14, 0.8), (1150.0, 0.18, 0.55), (2500.0, 0.09, 0.25)),
        "wall": ((120.0, 0.045, 1.0), (260.0, 0.030, 0.35), (650.0, 0.018, 0.12)),
    }
    modes = modes_by_material[material]
    spacing = int(interval * context.sample_rate)
    hit_size = int(max(decay for _, decay, _ in modes) * 5.0 * context.sample_rate)
    total = spacing * (count - 1) + hit_size
    result = np.zeros(total, dtype=np.float32)
    for index in range(count):
        rng = _rng(spec, index)
        hit = _resonant_impact(sample_rate=context.sample_rate, rng=rng, modes=modes)
        start = index * spacing
        level = force * float(rng.uniform(0.92, 1.05))
        result[start : start + hit.size] += hit * np.float32(level)
    return RenderedSound(_limit_peak(result), context.sample_rate, spec)


def footsteps(spec: SfxSpec, context: RenderContext) -> RenderedSound:
    params = spec.parameters
    surface = choice(params, "surface", "wood", {"wood", "stone", "gravel", "carpet"})
    footwear = choice(params, "footwear", "shoes", {"barefoot", "shoes", "boots", "heels"})
    count = integer(params, "count", 4, minimum=1, maximum=64)
    force = number(params, "force", 0.6, minimum=0.05, maximum=1.0)
    interval = number(params, "interval", 0.52, minimum=0.18, maximum=2.0)

    base_frequency = {"barefoot": 95.0, "shoes": 130.0, "boots": 105.0, "heels": 210.0}[footwear]
    surface_texture = {"wood": 0.08, "stone": 0.12, "gravel": 0.32, "carpet": 0.045}[surface]
    decay = {"wood": 0.055, "stone": 0.035, "gravel": 0.045, "carpet": 0.028}[surface]
    footwear_click = {"barefoot": 0.025, "shoes": 0.07, "boots": 0.095, "heels": 0.18}[footwear]
    hit_size = max(1, int(0.18 * context.sample_rate))
    t = np.arange(hit_size, dtype=np.float32) / np.float32(context.sample_rate)
    spacing = int(interval * context.sample_rate)
    total = spacing * (count - 1) + hit_size
    result = np.zeros(total, dtype=np.float32)

    for index in range(count):
        rng = _rng(spec, index)
        step_decay = decay * float(rng.uniform(0.9, 1.12))
        frequency = base_frequency * float(rng.uniform(0.94, 1.06))
        side = -1.0 if index % 2 else 1.0
        phase = float(rng.uniform(-0.12, 0.12))
        envelope = np.exp(-t / np.float32(step_decay))
        body = np.sin(np.float32(2.0 * math.pi * frequency) * t + np.float32(phase))
        body *= envelope * np.float32(0.67 + 0.025 * side)

        noise = rng.normal(0.0, 1.0, hit_size).astype(np.float32)
        high_component = noise - np.roll(noise, 1)
        high_component[0] = noise[0]
        contact_env = np.exp(-t / np.float32(0.0035))
        texture_env = np.exp(-t / np.float32(max(0.014, step_decay * 0.75)))
        contact = noise * contact_env * np.float32(0.18)
        texture = noise * texture_env * np.float32(surface_texture)
        texture += high_component * texture_env * np.float32(0.025)
        footwear_layer = high_component * contact_env * np.float32(footwear_click)
        if footwear == "heels":
            heel_freq = float(rng.uniform(1400.0, 1900.0))
            footwear_layer += np.sin(2.0 * math.pi * heel_freq * t) * contact_env * np.float32(0.12)

        hit = _normalize_template(body + contact + texture + footwear_layer, peak=0.72)
        if surface == "gravel":
            click_mask = rng.random(hit_size) < (28.0 / context.sample_rate)
            clicks = click_mask.astype(np.float32) * rng.uniform(0.2, 0.75, hit_size).astype(
                np.float32
            )
            hit += clicks * np.exp(-t / np.float32(0.012)) * np.float32(0.18)
        local_force = force * float(rng.uniform(0.94, 1.06))
        start = index * spacing
        result[start : start + hit_size] += hit * np.float32(local_force)

    result = _fade_out(result, context.sample_rate, 0.03)
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

"""Dependency-light procedural MVP effects."""

from __future__ import annotations

import math

import numpy as np

from ._params import choice, integer, number
from .types import FloatAudio, RenderContext, RenderedSound, SfxSpec


def _rng(spec: SfxSpec) -> np.random.Generator:
    return np.random.default_rng(spec.seed)


def _fade_out(signal: FloatAudio, sample_rate: int, seconds: float) -> FloatAudio:
    n = min(signal.size, max(1, int(seconds * sample_rate)))
    out = signal.copy()
    out[-n:] *= np.linspace(1.0, 0.0, n, dtype=np.float32)
    return out


def _normalize(signal: FloatAudio, peak: float = 0.92) -> FloatAudio:
    maximum = float(np.max(np.abs(signal))) if signal.size else 0.0
    if maximum > 0.0:
        signal = signal * np.float32(peak / maximum)
    return np.asarray(signal, dtype=np.float32)


def _resonant_impact(
    *, sample_rate: int, rng: np.random.Generator, force: float, modes: tuple[tuple[float, float, float], ...]
) -> FloatAudio:
    duration = max(decay for _, decay, _ in modes) * 5.0
    n = max(1, int(duration * sample_rate))
    t = np.arange(n, dtype=np.float32) / np.float32(sample_rate)
    signal = np.zeros(n, dtype=np.float32)
    phase = rng.uniform(-0.15, 0.15, size=len(modes))
    for index, (freq, decay, gain) in enumerate(modes):
        env = np.exp(-t / np.float32(decay))
        signal += np.float32(gain) * env * np.sin(
            np.float32(2.0 * math.pi * freq) * t + np.float32(phase[index])
        )
    transient_n = max(1, int(0.008 * sample_rate))
    transient = rng.normal(0.0, 1.0, transient_n).astype(np.float32)
    transient *= np.linspace(1.0, 0.0, transient_n, dtype=np.float32)
    signal[:transient_n] += np.float32(0.45) * transient
    return _normalize(signal * np.float32(force))


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
    rng = _rng(spec)
    hit = _resonant_impact(
        sample_rate=context.sample_rate, rng=rng, force=force, modes=modes_by_material[material]
    )
    spacing = int(interval * context.sample_rate)
    total = spacing * (count - 1) + hit.size
    result = np.zeros(total, dtype=np.float32)
    for index in range(count):
        start = index * spacing
        jitter = np.float32(rng.uniform(0.92, 1.05))
        result[start : start + hit.size] += hit * jitter
    return RenderedSound(_normalize(result), context.sample_rate, spec)


def footsteps(spec: SfxSpec, context: RenderContext) -> RenderedSound:
    params = spec.parameters
    surface = choice(params, "surface", "wood", {"wood", "stone", "gravel", "carpet"})
    footwear = choice(params, "footwear", "shoes", {"barefoot", "shoes", "boots", "heels"})
    count = integer(params, "count", 4, minimum=1, maximum=64)
    force = number(params, "force", 0.6, minimum=0.05, maximum=1.0)
    interval = number(params, "interval", 0.52, minimum=0.18, maximum=2.0)
    rng = _rng(spec)

    base_freq = {"barefoot": 95.0, "shoes": 130.0, "boots": 105.0, "heels": 210.0}[footwear]
    surface_noise = {"wood": 0.12, "stone": 0.10, "gravel": 0.42, "carpet": 0.05}[surface]
    decay = {"wood": 0.055, "stone": 0.035, "gravel": 0.045, "carpet": 0.028}[surface]
    hit_n = max(1, int(0.18 * context.sample_rate))
    t = np.arange(hit_n, dtype=np.float32) / np.float32(context.sample_rate)
    spacing = int(interval * context.sample_rate)
    total = spacing * (count - 1) + hit_n
    result = np.zeros(total, dtype=np.float32)

    for index in range(count):
        local_force = force * float(rng.uniform(0.86, 1.08))
        frequency = base_freq * float(rng.uniform(0.94, 1.06))
        env = np.exp(-t / np.float32(decay))
        body = np.sin(np.float32(2.0 * math.pi * frequency) * t) * env
        noise = rng.normal(0.0, 1.0, hit_n).astype(np.float32)
        noise_env = np.exp(-t / np.float32(max(0.015, decay * 0.65)))
        hit = (body * np.float32(0.65) + noise * noise_env * np.float32(surface_noise))
        if surface == "gravel":
            clicks = rng.random(hit_n) < (20.0 / context.sample_rate)
            hit += clicks.astype(np.float32) * rng.uniform(0.2, 0.8, hit_n).astype(np.float32)
        start = index * spacing
        result[start : start + hit_n] += hit * np.float32(local_force)
    return RenderedSound(_normalize(_fade_out(result, context.sample_rate, 0.03)), context.sample_rate, spec)


def phone_ring(spec: SfxSpec, context: RenderContext) -> RenderedSound:
    params = spec.parameters
    style = choice(params, "style", "classic", {"classic", "electronic"})
    count = integer(params, "count", 1, minimum=1, maximum=12)
    interval = number(params, "interval", 1.15, minimum=0.25, maximum=5.0)
    rng = _rng(spec)
    ring_duration = 0.62 if style == "classic" else 0.42
    n = int(ring_duration * context.sample_rate)
    t = np.arange(n, dtype=np.float32) / np.float32(context.sample_rate)
    if style == "classic":
        carrier = 0.62 * np.sin(2 * np.pi * 440.0 * t) + 0.38 * np.sin(2 * np.pi * 480.0 * t)
        wobble = 0.72 + 0.28 * np.sin(2 * np.pi * 20.0 * t)
        ring = carrier * wobble
    else:
        carrier = np.sin(2 * np.pi * 880.0 * t) + 0.45 * np.sin(2 * np.pi * 1320.0 * t)
        gate = (np.sin(2 * np.pi * 7.0 * t) > -0.15).astype(np.float32)
        ring = carrier * gate
    ring = np.asarray(ring, dtype=np.float32) * np.float32(rng.uniform(0.96, 1.0))
    ring = _fade_out(_normalize(ring), context.sample_rate, 0.04)
    spacing = int(interval * context.sample_rate)
    total = spacing * (count - 1) + ring.size
    result = np.zeros(total, dtype=np.float32)
    for index in range(count):
        start = index * spacing
        result[start : start + ring.size] += ring
    return RenderedSound(_normalize(result), context.sample_rate, spec)


def door_open(spec: SfxSpec, context: RenderContext) -> RenderedSound:
    params = spec.parameters
    material = choice(params, "material", "wood", {"wood", "metal"})
    speed = choice(params, "speed", "normal", {"slow", "normal", "fast"})
    creak = number(params, "creak", 0.65, minimum=0.0, maximum=1.0)
    duration = {"slow": 2.4, "normal": 1.5, "fast": 0.85}[speed]
    rng = _rng(spec)
    n = int(duration * context.sample_rate)
    t = np.arange(n, dtype=np.float32) / np.float32(context.sample_rate)
    start_f = 155.0 if material == "wood" else 260.0
    end_f = 380.0 if material == "wood" else 720.0
    phase = 2.0 * np.pi * (start_f * t + 0.5 * (end_f - start_f) / duration * t * t)
    envelope = np.sin(np.pi * np.clip(t / duration, 0.0, 1.0)) ** 1.4
    squeak = np.sin(phase).astype(np.float32) * envelope.astype(np.float32)
    texture = rng.normal(0.0, 1.0, n).astype(np.float32)
    texture *= (0.08 if material == "wood" else 0.05) * envelope.astype(np.float32)
    handle_n = min(n, int(0.09 * context.sample_rate))
    handle = np.zeros(n, dtype=np.float32)
    handle[:handle_n] = rng.normal(0.0, 1.0, handle_n).astype(np.float32) * np.linspace(
        0.45, 0.0, handle_n, dtype=np.float32
    )
    result = squeak * np.float32(creak) + texture + handle
    return RenderedSound(_normalize(_fade_out(result, context.sample_rate, 0.05)), context.sample_rate, spec)

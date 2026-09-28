"""Dependency-light procedural MVP effects."""

from __future__ import annotations
from ._doorbell import generate_door_chime, render_door_chime, render_electronic_doorbell
from ._phone import (
    generate_classic_phone_ringer,
    render_classic_phone_ring,
    render_electronic_phone_ring,
)

import math

import numpy as np

from ._doors import generate_door_model, render_close, render_open
from ._footsteps import aggregate_footstep, solid_footstep
from ._params import choice, integer, number
from ._pen import render_pen_write
from ._object_effects import (
    render_button_press,
    render_glass_clink,
    render_object_set_down,
    render_switch_toggle,
)
from ._paper import render_page_turn, render_paper_handle
from ._physics.contact import ImpactContact, impact_force
from ._physics.geometry import rectangular_plate_modes
from ._physics.physical_impacts import render_physical_impact
from ._physics.presets import KNOCK_IMPACTORS, KNOCK_OBJECTS, OBJECT_PRESETS
from ._physics.rng import RandomStream, event_rng
from ._printer import (
    generate_printer_model,
    render_printer_power_switch,
    render_printer_print,
    render_printer_restart,
    render_printer_tray_open,
    render_printer_tray_close,
    render_printer_wake,
)
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
    material_name: str,
    impactor_name: str,
    force: float,
    sample_rate: int,
    seed: int,
    event_index: int,
) -> FloatAudio:
    """Create one stable object-mode response to a compliant physical impact."""
    object_preset = OBJECT_PRESETS[KNOCK_OBJECTS[material_name]]
    impactor = KNOCK_IMPACTORS[impactor_name]
    position_rng = event_rng(seed, event_index, RandomStream.POSITION)
    variation_rng = event_rng(seed, event_index, RandomStream.VARIATION)
    centers = {
        "small_wood_board": (0.53, 0.48),
        "oak_door": (0.77, 0.54),
        "drywall_panel": (0.46, 0.56),
        "steel_sheet": (0.58, 0.52),
    }
    center_x, center_y = centers[object_preset.name]
    contact_x = float(np.clip(center_x + position_rng.normal(0.0, 0.045), 0.08, 0.92))
    contact_y = float(np.clip(center_y + position_rng.normal(0.0, 0.055), 0.08, 0.92))
    modes = rectangular_plate_modes(
        object_preset.geometry,
        object_preset.material,
        sample_rate=sample_rate,
        max_modes=32,
        contact_x=contact_x,
        contact_y=contact_y,
    )
    stiffness = impactor.stiffness * (0.55 + 0.65 * object_preset.material.contact_hardness)
    restitution = math.sqrt(impactor.restitution * object_preset.material.restitution)
    contact = ImpactContact(
        effective_mass_kg=impactor.effective_mass_kg,
        stiffness=stiffness,
        exponent=1.5,
        restitution=restitution,
    )
    velocity = (0.15 + 1.85 * force) * float(variation_rng.uniform(0.97, 1.03))
    trace = impact_force(
        contact=contact,
        velocity_m_s=velocity,
        sample_rate=sample_rate,
        oversample=4,
    )
    texture_rng = event_rng(seed, event_index, RandomStream.ROUGHNESS)
    body_mass = (
        object_preset.material.density_kg_m3
        * object_preset.geometry.width_m
        * object_preset.geometry.height_m
        * object_preset.geometry.thickness_m
    )
    output_gain = 20.0 + 12.0 / max(body_mass, 0.2)
    hit = render_physical_impact(
        trace,
        modes,
        sample_rate=sample_rate,
        rng=texture_rng,
        microscopic_gain=impactor.microscopic_gain
        * (0.5 + object_preset.material.roughness_rms_m / 5e-6),
        output_gain=output_gain,
    )
    minimum_size = round(0.20 * sample_rate) if object_preset.name == "small_wood_board" else 0
    if hit.size < minimum_size:
        hit = np.pad(hit, (0, minimum_size - hit.size)).astype(np.float32)
    return hit


def knock(spec: SfxSpec, context: RenderContext) -> RenderedSound:
    params = spec.parameters
    material = choice(params, "material", "wood", {"wood", "oak", "metal", "wall"})
    impactor = choice(params, "impactor", "knuckle", set(KNOCK_IMPACTORS))
    count = integer(params, "count", 1, minimum=1, maximum=16)
    force = number(params, "force", 0.65, minimum=0.05, maximum=1.0)
    interval = number(params, "interval", 0.22, minimum=0.08, maximum=2.0)
    seed = spec.seed if spec.seed is not None else 0
    starts = _event_starts(
        count=count,
        interval=interval,
        sample_rate=context.sample_rate,
        spec=spec,
        jitter_fraction=0.008,
    )
    hits = [
        _knock_hit(
            material_name=material,
            impactor_name=impactor,
            force=force,
            sample_rate=context.sample_rate,
            seed=seed,
            event_index=index,
        )
        for index in range(count)
    ]
    total = max(start + hit.size for start, hit in zip(starts, hits, strict=True))
    result = np.zeros(total, dtype=np.float32)
    for start, hit in zip(starts, hits, strict=True):
        _mix_at(result, hit, start)
    return RenderedSound(_limit_peak(result), context.sample_rate, spec)


def switch_toggle(spec: SfxSpec, context: RenderContext) -> RenderedSound:
    state = choice(spec.parameters, "state", "on", {"on", "off"})
    seed = spec.seed if spec.seed is not None else 0
    samples = render_switch_toggle(sample_rate=context.sample_rate, state=state, seed=seed)
    return RenderedSound(_limit_peak(samples), context.sample_rate, spec)


def button_press(spec: SfxSpec, context: RenderContext) -> RenderedSound:
    size = choice(spec.parameters, "size", "small", {"small", "large"})
    force = choice(spec.parameters, "force", "normal", {"gentle", "normal", "firm"})
    seed = spec.seed if spec.seed is not None else 0
    samples = render_button_press(
        sample_rate=context.sample_rate, size=size, force=force, seed=seed
    )
    return RenderedSound(_limit_peak(samples), context.sample_rate, spec)


def object_set_down(spec: SfxSpec, context: RenderContext) -> RenderedSound:
    object_type = choice(
        spec.parameters, "object", "wood", {"wood", "metal", "ceramic"}
    )
    surface = choice(spec.parameters, "surface", "wood", {"wood", "stone"})
    force = choice(spec.parameters, "force", "normal", {"gentle", "normal", "firm"})
    seed = spec.seed if spec.seed is not None else 0
    samples = render_object_set_down(
        sample_rate=context.sample_rate,
        object_type=object_type,
        surface=surface,
        force=force,
        seed=seed,
    )
    return RenderedSound(_limit_peak(samples), context.sample_rate, spec)


def glass_clink(spec: SfxSpec, context: RenderContext) -> RenderedSound:
    style = choice(spec.parameters, "style", "wine", {"wine", "tumbler"})
    count = integer(spec.parameters, "count", 1, minimum=1, maximum=4)
    force = choice(spec.parameters, "force", "normal", {"gentle", "normal", "firm"})
    seed = spec.seed if spec.seed is not None else 0
    samples = render_glass_clink(
        sample_rate=context.sample_rate, style=style, count=count, force=force, seed=seed
    )
    return RenderedSound(_limit_peak(samples), context.sample_rate, spec)


def paper_page_turn(spec: SfxSpec, context: RenderContext) -> RenderedSound:
    pages = integer(spec.parameters, "pages", 1, minimum=1, maximum=6)
    speed = choice(spec.parameters, "speed", "normal", {"slow", "normal", "fast"})
    seed = spec.seed if spec.seed is not None else 0
    samples = render_page_turn(
        sample_rate=context.sample_rate, pages=pages, speed=speed, seed=seed
    )
    return RenderedSound(_limit_peak(samples), context.sample_rate, spec)


def paper_handle(spec: SfxSpec, context: RenderContext) -> RenderedSound:
    duration = number(
        spec.parameters, "duration", 1.2, minimum=0.3, maximum=4.0
    )
    intensity = choice(
        spec.parameters, "intensity", "normal", {"gentle", "normal", "rough"}
    )
    seed = spec.seed if spec.seed is not None else 0
    samples = render_paper_handle(
        sample_rate=context.sample_rate,
        duration=duration,
        intensity=intensity,
        seed=seed,
    )
    return RenderedSound(_limit_peak(samples), context.sample_rate, spec)

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
    seed = spec.seed if spec.seed is not None else 0
    ring_duration = 0.62 if style == "classic" else 0.42
    if style == "classic":
        model = generate_classic_phone_ringer(seed=seed, sample_rate=context.sample_rate)
        ring = render_classic_phone_ring(
            model, duration_s=ring_duration, sample_rate=context.sample_rate
        )
    else:
        ring = render_electronic_phone_ring(
            seed=seed, duration_s=ring_duration, sample_rate=context.sample_rate
        )
    spacing = int(interval * context.sample_rate)
    total = spacing * (count - 1) + ring.size
    result = np.zeros(total, dtype=np.float32)
    for index in range(count):
        _mix_at(result, ring, index * spacing)
    return RenderedSound(_limit_peak(result), context.sample_rate, spec)



def doorbell_ring(spec: SfxSpec, context: RenderContext) -> RenderedSound:
    params = spec.parameters
    style = choice(params, "style", "chime", {"chime", "electronic"})
    count = integer(params, "count", 1, minimum=1, maximum=12)
    interval = number(params, "interval", 0.8, minimum=0.2, maximum=10.0)
    seed = spec.seed if spec.seed is not None else 0
    ring_duration = 0.72
    if style == "chime":
        model = generate_door_chime(seed=seed, sample_rate=context.sample_rate)
        ring = render_door_chime(
            model, duration_s=ring_duration, sample_rate=context.sample_rate
        )
    else:
        ring = render_electronic_doorbell(
            seed=seed, duration_s=ring_duration, sample_rate=context.sample_rate
        )
    spacing = int(interval * context.sample_rate)
    total = spacing * (count - 1) + ring.size
    result = np.zeros(total, dtype=np.float32)
    for index in range(count):
        _mix_at(result, ring, index * spacing)
    return RenderedSound(_limit_peak(result), context.sample_rate, spec)

def door_open(spec: SfxSpec, context: RenderContext) -> RenderedSound:
    params = spec.parameters
    material = choice(params, "material", "wood", {"wood", "metal"})
    speed = choice(params, "speed", "normal", {"slow", "normal", "fast"})
    creak = number(params, "creak", 0.65, minimum=0.0, maximum=1.0)
    force = number(params, "force", 0.45, minimum=0.05, maximum=1.0)
    seed = spec.seed if spec.seed is not None else 0
    model = generate_door_model(material=material, seed=seed, sample_rate=context.sample_rate)
    samples = render_open(
        model=model,
        speed=speed,
        creak=creak,
        force=force,
        sample_rate=context.sample_rate,
    )
    return RenderedSound(samples, context.sample_rate, spec)


def door_close(spec: SfxSpec, context: RenderContext) -> RenderedSound:
    params = spec.parameters
    material = choice(params, "material", "wood", {"wood", "metal"})
    speed = choice(params, "speed", "normal", {"slow", "normal", "fast"})
    creak = number(params, "creak", 0.25, minimum=0.0, maximum=1.0)
    force = number(params, "force", 0.65, minimum=0.05, maximum=1.0)
    seed = spec.seed if spec.seed is not None else 0
    model = generate_door_model(material=material, seed=seed, sample_rate=context.sample_rate)
    samples = render_close(
        model=model,
        speed=speed,
        creak=creak,
        force=force,
        sample_rate=context.sample_rate,
    )
    return RenderedSound(samples, context.sample_rate, spec)


def printer_print(spec: SfxSpec, context: RenderContext) -> RenderedSound:
    params = spec.parameters
    pages = integer(params, "pages", 1, minimum=1, maximum=12)
    speed = choice(params, "speed", "normal", {"slow", "normal", "fast"})
    seed = spec.seed if spec.seed is not None else 0
    samples = render_printer_print(
        sample_rate=context.sample_rate, pages=pages, speed=speed, seed=seed
    )
    return RenderedSound(_limit_peak(samples), context.sample_rate, spec)


def printer_tray_open(spec: SfxSpec, context: RenderContext) -> RenderedSound:
    params = spec.parameters
    speed = choice(params, "speed", "normal", {"slow", "normal", "fast"})
    paper_load = choice(params, "paper_load", "full", {"empty", "partial", "full"})
    seed = spec.seed if spec.seed is not None else 0
    model = generate_printer_model(seed=seed, sample_rate=context.sample_rate)
    samples = render_printer_tray_open(
        sample_rate=context.sample_rate,
        speed=speed,
        paper_load=paper_load,
        seed=seed,
        model=model,
    )
    return RenderedSound(_limit_peak(samples), context.sample_rate, spec)


def printer_tray_close(spec: SfxSpec, context: RenderContext) -> RenderedSound:
    params = spec.parameters
    speed = choice(params, "speed", "normal", {"slow", "normal", "fast"})
    paper_load = choice(params, "paper_load", "full", {"empty", "partial", "full"})
    force = choice(params, "force", "normal", {"gentle", "normal", "firm"})
    seed = spec.seed if spec.seed is not None else 0
    model = generate_printer_model(seed=seed, sample_rate=context.sample_rate)
    samples = render_printer_tray_close(
        sample_rate=context.sample_rate,
        speed=speed,
        paper_load=paper_load,
        force=force,
        seed=seed,
        model=model,
    )
    return RenderedSound(_limit_peak(samples), context.sample_rate, spec)


def printer_power_switch(spec: SfxSpec, context: RenderContext) -> RenderedSound:
    params = spec.parameters
    state = choice(params, "state", "off", {"off", "on"})
    seed = spec.seed if spec.seed is not None else 0
    samples = render_printer_power_switch(sample_rate=context.sample_rate, state=state, seed=seed)
    return RenderedSound(_limit_peak(samples), context.sample_rate, spec)


def printer_restart(spec: SfxSpec, context: RenderContext) -> RenderedSound:
    params = spec.parameters
    speed = choice(params, "speed", "normal", {"slow", "normal", "fast"})
    beep = choice(params, "beep", "off", {"off", "on"}) == "on"
    seed = spec.seed if spec.seed is not None else 0
    samples = render_printer_restart(
        sample_rate=context.sample_rate, speed=speed, seed=seed, beep=beep
    )
    return RenderedSound(_limit_peak(samples), context.sample_rate, spec)


def pen_write(spec: SfxSpec, context: RenderContext) -> RenderedSound:
    params = spec.parameters
    duration = number(params, "duration", 1.6, minimum=0.25, maximum=8.0)
    speed = choice(params, "speed", "normal", {"slow", "normal", "fast"})
    pressure = number(params, "pressure", 0.55, minimum=0.1, maximum=1.0)
    seed = spec.seed if spec.seed is not None else 0
    samples = render_pen_write(
        sample_rate=context.sample_rate,
        duration=duration,
        speed=speed,
        pressure=pressure,
        seed=seed,
    )
    return RenderedSound(_limit_peak(samples), context.sample_rate, spec)


def printer_wake(spec: SfxSpec, context: RenderContext) -> RenderedSound:
    params = spec.parameters
    depth = choice(params, "depth", "light", {"light", "deep"})
    beep = choice(params, "beep", "off", {"off", "on"}) == "on"
    seed = spec.seed if spec.seed is not None else 0
    samples = render_printer_wake(
        sample_rate=context.sample_rate, depth=depth, seed=seed, beep=beep
    )
    return RenderedSound(_limit_peak(samples), context.sample_rate, spec)

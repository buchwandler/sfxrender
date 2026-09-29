"""Dependency-light procedural MVP effects."""

from __future__ import annotations

import math

import numpy as np

from ._ambience import (
    render_city_ambience,
    render_office_ambience,
    render_room_tone,
)
from ._device_effects import (
    render_device_beep,
    render_device_power_off,
    render_device_power_on,
    render_electronics_hum,
    render_phone_notification,
    render_phone_vibrate,
)
from ._doorbell import generate_door_chime, render_door_chime, render_electronic_doorbell
from ._doors import generate_door_model, render_close, render_open
from ._environment import (
    render_birds_ambience,
    render_crickets_ambience,
    render_fire_crackle,
    render_rain,
    render_transition_whoosh,
    render_wind,
)
from ._fluid_effects import (
    render_crowd_murmur,
    render_thunder,
    render_water_pour,
    render_water_running,
)
from ._footsteps import aggregate_footstep, solid_footstep
from ._keyboard import render_keyboard_typing
from ._material_effects import (
    render_car_door,
    render_chair_move,
    render_cloth_rustle,
    render_floor_creak,
    render_lock_turn,
)
from ._metal_effects import (
    render_alarm_ring,
    render_clock_tick,
    render_keys_jingle,
)
from ._object_effects import (
    render_button_press,
    render_glass_clink,
    render_object_set_down,
    render_switch_toggle,
)
from ._paper import render_page_turn, render_paper_handle
from ._params import choice, integer, number
from ._pen import render_pen_write
from ._phone import (
    generate_classic_phone_ringer,
    render_classic_phone_ring,
    render_electronic_phone_ring,
)
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
    render_printer_tray_close,
    render_printer_tray_open,
    render_printer_wake,
)
from ._transport_effects import (
    render_car_engine,
    render_car_passby,
    render_elevator_arrive,
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
    output_gain = 2.0 + 1.2 / max(body_mass, 0.2)
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
    object_type = choice(spec.parameters, "object", "wood", {"wood", "metal", "ceramic"})
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
    samples = render_page_turn(sample_rate=context.sample_rate, pages=pages, speed=speed, seed=seed)
    return RenderedSound(_limit_peak(samples), context.sample_rate, spec)


def paper_handle(spec: SfxSpec, context: RenderContext) -> RenderedSound:
    duration = number(spec.parameters, "duration", 1.2, minimum=0.3, maximum=4.0)
    intensity = choice(spec.parameters, "intensity", "normal", {"gentle", "normal", "rough"})
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
    gait: str = "walk",
    step_interval_s: float,
) -> FloatAudio:
    """Route one pace-aware event through the shared foot-ground model."""
    side_variation = 0.09 if gait == "run" else 0.03
    local_force = force * (1.0 + side_variation * side)
    if surface == "gravel":
        return aggregate_footstep(
            sample_rate=sample_rate,
            surface=surface,
            footwear=footwear,
            force=local_force,
            rng=rng,
            gait=gait,
            step_interval_s=step_interval_s,
        )
    return solid_footstep(
        sample_rate=sample_rate,
        surface=surface,
        footwear=footwear,
        force=local_force,
        rng=rng,
        gait=gait,
        step_interval_s=step_interval_s,
    )


def _render_footstep_sequence(
    spec: SfxSpec,
    context: RenderContext,
    *,
    gait: str,
    default_count: int,
    default_force: float,
    default_interval: float,
    minimum_interval: float,
    maximum_interval: float,
    jitter_fraction: float,
) -> RenderedSound:
    params = spec.parameters
    surface_values = (
        {"wood", "stone"} if gait.startswith("stairs_") else {"wood", "stone", "gravel", "carpet"}
    )
    surface = choice(params, "surface", "wood", surface_values)
    footwear = choice(params, "footwear", "shoes", {"barefoot", "shoes", "boots", "heels"})
    count = integer(params, "count", default_count, minimum=1, maximum=64)
    force = number(params, "force", default_force, minimum=0.05, maximum=1.0)
    interval = number(
        params,
        "interval",
        default_interval,
        minimum=minimum_interval,
        maximum=maximum_interval,
    )
    starts = _event_starts(
        count=count,
        interval=interval,
        sample_rate=context.sample_rate,
        spec=spec,
        jitter_fraction=jitter_fraction,
    )
    gait_force = force * (1.18 if gait == "run" else 1.0)
    hits: list[FloatAudio] = []
    gains: list[float] = []
    for index in range(count):
        rng = _rng(spec, index)
        side = -1.0 if index % 2 else 1.0
        hit = _footstep_hit(
            sample_rate=context.sample_rate,
            surface=surface,
            footwear=footwear,
            rng=rng,
            force=gait_force,
            side=side,
            gait=gait,
            step_interval_s=interval,
        )
        variation = 0.15 if gait == "run" else 0.08 if gait.startswith("stairs_") else 0.05
        side_gain = 0.09 if gait == "run" else 0.06 if gait.startswith("stairs_") else 0.025
        gains.append(
            float(rng.uniform(1.0 - variation, 1.0 + variation)) * (1.0 + side_gain * side)
        )
        hits.append(hit)
    if gait == "walk":
        duration = {"barefoot": 0.292, "shoes": 0.265, "boots": 0.305, "heels": 0.245}[footwear]
        if surface == "gravel":
            duration += 0.02
        output_size = starts[-1] + max(1, round(duration * context.sample_rate))
    else:
        output_size = max(start + hit.size for start, hit in zip(starts, hits, strict=True))
    result = np.zeros(output_size, dtype=np.float32)
    for start, hit, gain in zip(starts, hits, gains, strict=True):
        _mix_at(result, hit * np.float32(gain), start)
    result = _fade_out(result, context.sample_rate, 0.025)
    return RenderedSound(_limit_peak(result), context.sample_rate, spec)


def footsteps(spec: SfxSpec, context: RenderContext) -> RenderedSound:
    return _render_footstep_sequence(
        spec,
        context,
        gait="walk",
        default_count=4,
        default_force=0.6,
        default_interval=0.52,
        minimum_interval=0.18,
        maximum_interval=2.0,
        jitter_fraction=0.04,
    )


def footsteps_run(spec: SfxSpec, context: RenderContext) -> RenderedSound:
    return _render_footstep_sequence(
        spec,
        context,
        gait="run",
        default_count=8,
        default_force=0.68,
        default_interval=0.32,
        minimum_interval=0.20,
        maximum_interval=0.8,
        jitter_fraction=0.08,
    )


def footsteps_stairs(spec: SfxSpec, context: RenderContext) -> RenderedSound:
    direction = choice(spec.parameters, "direction", "up", {"up", "down"})
    interval = 0.58
    return _render_footstep_sequence(
        spec,
        context,
        gait=f"stairs_{direction}",
        default_count=5,
        default_force=0.62,
        default_interval=interval,
        minimum_interval=0.28,
        maximum_interval=1.6,
        jitter_fraction=0.05,
    )


def electronics_hum(spec: SfxSpec, context: RenderContext) -> RenderedSound:
    source = choice(spec.parameters, "source", "transformer", {"transformer", "appliance"})
    duration = number(spec.parameters, "duration", 2.0, minimum=0.5, maximum=8.0)
    seed = spec.seed if spec.seed is not None else 0
    samples = render_electronics_hum(
        sample_rate=context.sample_rate, duration=duration, device=source, seed=seed
    )
    return RenderedSound(_limit_peak(samples), context.sample_rate, spec)


def device_beep(spec: SfxSpec, context: RenderContext) -> RenderedSound:
    style = choice(spec.parameters, "style", "soft", {"soft", "alert"})
    pattern = choice(spec.parameters, "pattern", "single", {"single", "double", "triple"})
    seed = spec.seed if spec.seed is not None else 0
    samples = render_device_beep(
        sample_rate=context.sample_rate, style=style, pattern=pattern, seed=seed
    )
    return RenderedSound(_limit_peak(samples), context.sample_rate, spec)


def phone_notification(spec: SfxSpec, context: RenderContext) -> RenderedSound:
    style = choice(spec.parameters, "style", "gentle", {"gentle", "urgent"})
    count = integer(spec.parameters, "count", 1, minimum=1, maximum=4)
    seed = spec.seed if spec.seed is not None else 0
    samples = render_phone_notification(
        sample_rate=context.sample_rate, style=style, count=count, seed=seed
    )
    return RenderedSound(_limit_peak(samples), context.sample_rate, spec)


def phone_vibrate(spec: SfxSpec, context: RenderContext) -> RenderedSound:
    duration = number(spec.parameters, "duration", 0.8, minimum=0.2, maximum=5.0)
    intensity = choice(spec.parameters, "intensity", "normal", {"gentle", "normal", "strong"})
    pattern = choice(spec.parameters, "pattern", "steady", {"steady", "pulsed"})
    surface = choice(spec.parameters, "surface", "wood", {"wood", "stone"})
    seed = spec.seed if spec.seed is not None else 0
    samples = render_phone_vibrate(
        sample_rate=context.sample_rate,
        duration=duration,
        intensity=intensity,
        pattern=pattern,
        surface=surface,
        seed=seed,
    )
    return RenderedSound(_limit_peak(samples), context.sample_rate, spec)


def device_power_on(spec: SfxSpec, context: RenderContext) -> RenderedSound:
    device = choice(spec.parameters, "device", "small", {"small", "appliance"})
    seed = spec.seed if spec.seed is not None else 0
    samples = render_device_power_on(sample_rate=context.sample_rate, device=device, seed=seed)
    return RenderedSound(_limit_peak(samples), context.sample_rate, spec)


def device_power_off(spec: SfxSpec, context: RenderContext) -> RenderedSound:
    device = choice(spec.parameters, "device", "small", {"small", "appliance"})
    seed = spec.seed if spec.seed is not None else 0
    samples = render_device_power_off(sample_rate=context.sample_rate, device=device, seed=seed)
    return RenderedSound(_limit_peak(samples), context.sample_rate, spec)


def room_tone(spec: SfxSpec, context: RenderContext) -> RenderedSound:
    character = choice(
        spec.parameters,
        "character",
        "quiet",
        {"quiet", "ventilated", "electrical"},
    )
    duration = number(spec.parameters, "duration", 4.0, minimum=0.5, maximum=30.0)
    seed = spec.seed if spec.seed is not None else 0
    samples = render_room_tone(
        sample_rate=context.sample_rate,
        duration=duration,
        character=character,
        seed=seed,
    )
    return RenderedSound(_limit_peak(samples), context.sample_rate, spec)


def office_ambience(spec: SfxSpec, context: RenderContext) -> RenderedSound:
    activity = choice(spec.parameters, "activity", "quiet", {"quiet", "busy"})
    duration = number(spec.parameters, "duration", 8.0, minimum=2.0, maximum=30.0)
    seed = spec.seed if spec.seed is not None else 0
    samples = render_office_ambience(
        sample_rate=context.sample_rate,
        duration=duration,
        activity=activity,
        seed=seed,
    )
    return RenderedSound(_limit_peak(samples), context.sample_rate, spec)


def city_ambience(spec: SfxSpec, context: RenderContext) -> RenderedSound:
    activity = choice(spec.parameters, "activity", "calm", {"calm", "busy"})
    duration = number(spec.parameters, "duration", 12.0, minimum=4.0, maximum=30.0)
    seed = spec.seed if spec.seed is not None else 0
    samples = render_city_ambience(
        sample_rate=context.sample_rate,
        duration=duration,
        activity=activity,
        seed=seed,
    )
    return RenderedSound(_limit_peak(samples), context.sample_rate, spec)


def wind(spec: SfxSpec, context: RenderContext) -> RenderedSound:
    intensity = choice(spec.parameters, "intensity", "light", {"light", "strong"})
    texture = choice(spec.parameters, "texture", "smooth", {"smooth", "leafy"})
    duration = number(spec.parameters, "duration", 6.0, minimum=1.0, maximum=30.0)
    seed = spec.seed if spec.seed is not None else 0
    samples = render_wind(
        sample_rate=context.sample_rate,
        duration=duration,
        intensity=intensity,
        texture=texture,
        seed=seed,
    )
    return RenderedSound(_limit_peak(samples), context.sample_rate, spec)


def transition_whoosh(spec: SfxSpec, context: RenderContext) -> RenderedSound:
    style = choice(spec.parameters, "style", "soft", {"soft", "forceful"})
    direction = choice(spec.parameters, "direction", "rise", {"rise", "fall"})
    duration = number(spec.parameters, "duration", 1.2, minimum=0.25, maximum=4.0)
    seed = spec.seed if spec.seed is not None else 0
    samples = render_transition_whoosh(
        sample_rate=context.sample_rate,
        duration=duration,
        style=style,
        direction=direction,
        seed=seed,
    )
    return RenderedSound(_limit_peak(samples), context.sample_rate, spec)


def rain(spec: SfxSpec, context: RenderContext) -> RenderedSound:
    intensity = choice(spec.parameters, "intensity", "steady", {"light", "steady", "heavy"})
    surface = choice(spec.parameters, "surface", "ground", {"ground", "roof", "window"})
    duration = number(spec.parameters, "duration", 8.0, minimum=1.0, maximum=30.0)
    seed = spec.seed if spec.seed is not None else 0
    samples = render_rain(
        sample_rate=context.sample_rate,
        duration=duration,
        intensity=intensity,
        surface=surface,
        seed=seed,
    )
    return RenderedSound(_limit_peak(samples), context.sample_rate, spec)


def fire_crackle(spec: SfxSpec, context: RenderContext) -> RenderedSound:
    activity = choice(spec.parameters, "activity", "quiet", {"quiet", "active"})
    duration = number(spec.parameters, "duration", 8.0, minimum=2.0, maximum=30.0)
    seed = spec.seed if spec.seed is not None else 0
    samples = render_fire_crackle(
        sample_rate=context.sample_rate,
        duration=duration,
        activity=activity,
        seed=seed,
    )
    return RenderedSound(_limit_peak(samples), context.sample_rate, spec)


def birds_ambience(spec: SfxSpec, context: RenderContext) -> RenderedSound:
    activity = choice(spec.parameters, "activity", "sparse", {"sparse", "busy"})
    duration = number(spec.parameters, "duration", 12.0, minimum=2.0, maximum=30.0)
    seed = spec.seed if spec.seed is not None else 0
    samples = render_birds_ambience(
        sample_rate=context.sample_rate,
        duration=duration,
        activity=activity,
        seed=seed,
    )
    return RenderedSound(_limit_peak(samples), context.sample_rate, spec)


def crickets_ambience(spec: SfxSpec, context: RenderContext) -> RenderedSound:
    activity = choice(spec.parameters, "activity", "sparse", {"sparse", "busy"})
    duration = number(spec.parameters, "duration", 12.0, minimum=2.0, maximum=30.0)
    seed = spec.seed if spec.seed is not None else 0
    samples = render_crickets_ambience(
        sample_rate=context.sample_rate,
        duration=duration,
        activity=activity,
        seed=seed,
    )
    return RenderedSound(_limit_peak(samples), context.sample_rate, spec)


def keys_jingle(spec: SfxSpec, context: RenderContext) -> RenderedSound:
    style = choice(spec.parameters, "style", "light", {"light", "full"})
    duration = number(spec.parameters, "duration", 1.2, minimum=0.25, maximum=5.0)
    seed = spec.seed if spec.seed is not None else 0
    samples = render_keys_jingle(
        sample_rate=context.sample_rate, duration=duration, style=style, seed=seed
    )
    return RenderedSound(_limit_peak(samples), context.sample_rate, spec)


def lock_turn(spec: SfxSpec, context: RenderContext) -> RenderedSound:
    style = choice(spec.parameters, "style", "key", {"key", "deadbolt"})
    force = choice(spec.parameters, "force", "light", {"light", "firm"})
    duration = number(spec.parameters, "duration", 1.5, minimum=0.3, maximum=5.0)
    seed = spec.seed if spec.seed is not None else 0
    samples = render_lock_turn(
        sample_rate=context.sample_rate, duration=duration, style=style, force=force, seed=seed
    )
    return RenderedSound(_limit_peak(samples), context.sample_rate, spec)


def chair_move(spec: SfxSpec, context: RenderContext) -> RenderedSound:
    surface = choice(spec.parameters, "surface", "wood", {"wood", "stone", "carpet"})
    effort = choice(spec.parameters, "effort", "light", {"light", "firm"})
    action = choice(spec.parameters, "action", "slide", {"slide", "set_down"})
    duration = number(spec.parameters, "duration", 2.0, minimum=0.5, maximum=8.0)
    seed = spec.seed if spec.seed is not None else 0
    samples = render_chair_move(
        sample_rate=context.sample_rate,
        duration=duration,
        surface=surface,
        effort=effort,
        action=action,
        seed=seed,
    )
    return RenderedSound(_limit_peak(samples), context.sample_rate, spec)


def cloth_rustle(spec: SfxSpec, context: RenderContext) -> RenderedSound:
    fabric = choice(spec.parameters, "fabric", "cotton", {"cotton", "silk", "nylon"})
    activity = choice(spec.parameters, "activity", "light", {"light", "active"})
    duration = number(spec.parameters, "duration", 4.0, minimum=0.5, maximum=12.0)
    seed = spec.seed if spec.seed is not None else 0
    samples = render_cloth_rustle(
        sample_rate=context.sample_rate,
        duration=duration,
        fabric=fabric,
        activity_level=activity,
        seed=seed,
    )
    return RenderedSound(_limit_peak(samples), context.sample_rate, spec)


def floor_creak(spec: SfxSpec, context: RenderContext) -> RenderedSound:
    surface = choice(spec.parameters, "surface", "wood", {"wood", "carpet"})
    weight = choice(spec.parameters, "weight", "light", {"light", "heavy"})
    duration = number(spec.parameters, "duration", 1.3, minimum=0.4, maximum=4.0)
    seed = spec.seed if spec.seed is not None else 0
    samples = render_floor_creak(
        sample_rate=context.sample_rate,
        duration=duration,
        surface=surface,
        weight=weight,
        seed=seed,
    )
    return RenderedSound(_limit_peak(samples), context.sample_rate, spec)


def keyboard_typing(spec: SfxSpec, context: RenderContext) -> RenderedSound:
    speed = choice(spec.parameters, "speed", "steady", {"slow", "steady", "fast"})
    force = choice(spec.parameters, "force", "light", {"light", "firm"})
    duration = number(spec.parameters, "duration", 6.0, minimum=1.0, maximum=30.0)
    seed = spec.seed if spec.seed is not None else 0
    samples = render_keyboard_typing(
        sample_rate=context.sample_rate, duration=duration, speed=speed, force=force, seed=seed
    )
    return RenderedSound(_limit_peak(samples), context.sample_rate, spec)


def clock_tick(spec: SfxSpec, context: RenderContext) -> RenderedSound:
    style = choice(spec.parameters, "style", "wall", {"wall", "mantel"})
    rate = choice(spec.parameters, "rate", "normal", {"slow", "normal"})
    duration = number(spec.parameters, "duration", 6.0, minimum=1.0, maximum=30.0)
    seed = spec.seed if spec.seed is not None else 0
    samples = render_clock_tick(
        sample_rate=context.sample_rate, duration=duration, style=style, rate=rate, seed=seed
    )
    return RenderedSound(_limit_peak(samples), context.sample_rate, spec)


def alarm_ring(spec: SfxSpec, context: RenderContext) -> RenderedSound:
    style = choice(spec.parameters, "style", "mechanical", {"mechanical", "electronic"})
    pattern = choice(spec.parameters, "pattern", "intermittent", {"intermittent", "continuous"})
    duration = number(spec.parameters, "duration", 5.0, minimum=1.0, maximum=15.0)
    seed = spec.seed if spec.seed is not None else 0
    samples = render_alarm_ring(
        sample_rate=context.sample_rate, duration=duration, style=style, pattern=pattern, seed=seed
    )
    return RenderedSound(_limit_peak(samples), context.sample_rate, spec)


def elevator_arrive(spec: SfxSpec, context: RenderContext) -> RenderedSound:
    size = choice(spec.parameters, "size", "small", {"small", "large"})
    chime = choice(spec.parameters, "chime", "single", {"single", "double"})
    duration = number(spec.parameters, "duration", 3.5, minimum=1.0, maximum=8.0)
    seed = spec.seed if spec.seed is not None else 0
    samples = render_elevator_arrive(
        sample_rate=context.sample_rate, duration=duration, size=size, chime=chime, seed=seed
    )
    return RenderedSound(_limit_peak(samples), context.sample_rate, spec)


def water_pour(spec: SfxSpec, context: RenderContext) -> RenderedSound:
    flow = choice(spec.parameters, "flow", "steady", {"trickle", "steady", "strong"})
    vessel = choice(spec.parameters, "vessel", "glass", {"glass", "ceramic", "metal"})
    duration = number(spec.parameters, "duration", 4.0, minimum=0.5, maximum=12.0)
    seed = spec.seed if spec.seed is not None else 0
    samples = render_water_pour(
        sample_rate=context.sample_rate, duration=duration, flow=flow, vessel=vessel, seed=seed
    )
    return RenderedSound(_limit_peak(samples), context.sample_rate, spec)


def water_running(spec: SfxSpec, context: RenderContext) -> RenderedSound:
    flow = choice(spec.parameters, "flow", "steady", {"gentle", "steady", "strong"})
    duration = number(spec.parameters, "duration", 6.0, minimum=1.0, maximum=20.0)
    seed = spec.seed if spec.seed is not None else 0
    samples = render_water_running(
        sample_rate=context.sample_rate, duration=duration, flow=flow, seed=seed
    )
    return RenderedSound(_limit_peak(samples), context.sample_rate, spec)


def crowd_murmur(spec: SfxSpec, context: RenderContext) -> RenderedSound:
    density = choice(spec.parameters, "density", "moderate", {"sparse", "moderate", "busy"})
    duration = number(spec.parameters, "duration", 8.0, minimum=2.0, maximum=30.0)
    seed = spec.seed if spec.seed is not None else 0
    samples = render_crowd_murmur(
        sample_rate=context.sample_rate, duration=duration, density=density, seed=seed
    )
    return RenderedSound(_limit_peak(samples), context.sample_rate, spec)


def thunder(spec: SfxSpec, context: RenderContext) -> RenderedSound:
    intensity = choice(spec.parameters, "intensity", "strong", {"light", "strong"})
    distance = choice(spec.parameters, "distance", "distant", {"near", "distant"})
    duration = number(spec.parameters, "duration", 6.0, minimum=3.0, maximum=15.0)
    seed = spec.seed if spec.seed is not None else 0
    samples = render_thunder(
        sample_rate=context.sample_rate,
        duration=duration,
        intensity=intensity,
        distance=distance,
        seed=seed,
    )
    return RenderedSound(_limit_peak(samples), context.sample_rate, spec)


def car_door(spec: SfxSpec, context: RenderContext) -> RenderedSound:
    action = choice(spec.parameters, "action", "close", {"open", "close"})
    size = choice(spec.parameters, "size", "sedan", {"sedan", "suv"})
    force = choice(spec.parameters, "force", "firm", {"light", "firm"})
    duration = number(spec.parameters, "duration", 1.5, minimum=0.5, maximum=4.0)
    seed = spec.seed if spec.seed is not None else 0
    samples = render_car_door(
        sample_rate=context.sample_rate,
        duration=duration,
        action=action,
        size=size,
        force=force,
        seed=seed,
    )
    return RenderedSound(_limit_peak(samples), context.sample_rate, spec)


def car_engine(spec: SfxSpec, context: RenderContext) -> RenderedSound:
    action = choice(spec.parameters, "action", "idle", {"start", "idle", "rev"})
    vehicle = choice(spec.parameters, "vehicle", "compact", {"compact", "truck"})
    duration = number(spec.parameters, "duration", 5.0, minimum=1.0, maximum=15.0)
    seed = spec.seed if spec.seed is not None else 0
    samples = render_car_engine(
        sample_rate=context.sample_rate,
        duration=duration,
        action=action,
        vehicle=vehicle,
        seed=seed,
    )
    return RenderedSound(_limit_peak(samples), context.sample_rate, spec)


def car_passby(spec: SfxSpec, context: RenderContext) -> RenderedSound:
    speed = choice(spec.parameters, "speed", "slow", {"slow", "fast"})
    vehicle = choice(spec.parameters, "vehicle", "compact", {"compact", "truck"})
    duration = number(spec.parameters, "duration", 5.0, minimum=2.0, maximum=10.0)
    seed = spec.seed if spec.seed is not None else 0
    samples = render_car_passby(
        sample_rate=context.sample_rate, duration=duration, speed=speed, vehicle=vehicle, seed=seed
    )
    return RenderedSound(_limit_peak(samples), context.sample_rate, spec)


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
        ring = render_door_chime(model, duration_s=ring_duration, sample_rate=context.sample_rate)
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

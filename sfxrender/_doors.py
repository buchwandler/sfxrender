"""Generated door identity and action-independent physical properties."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

from ._door_profiles import DOOR_MATERIAL_PROFILES, DoorMaterialProfile
from ._physics.friction import friction
from ._physics.impacts import impact
from ._physics.models import ContactProfile, FrictionProfile, ModalBody, ModeBand
from ._physics.motion import eased_motion
from ._physics.resonators import generate_modal_body
from .types import FloatAudio

_LAYER_IDS: dict[str, int] = {
    "panel": 100,
    "frame": 200,
    "panel-properties": 110,
    "damping": 120,
    "latch": 300,
    "hinge": 400,
    "regions": 450,
    "handle": 500,
    "latch-contact": 600,
    "frame-contact": 700,
    "motion-open": 800,
    "motion-close": 900,
    "handle-event": 1_000,
    "latch-event": 1_100,
    "breakaway": 1_200,
    "hinge-event": 1_300,
    "terminal-stop": 1_400,
    "frame-impact": 1_500,
    "latch-catch": 1_600,
    "chatter": 1_700,
}


def door_rng(seed: int, component: str, event_index: int = 0) -> np.random.Generator:
    """Return a stable random substream; names use explicit IDs, never ``hash``."""
    if seed < 0 or event_index < 0:
        raise ValueError("seed and event_index must be non-negative")
    try:
        layer_id = _LAYER_IDS[component]
    except KeyError as exc:
        raise ValueError(f"unknown deterministic door component {component!r}") from exc
    return np.random.default_rng(np.random.SeedSequence([seed, layer_id, event_index]))


@dataclass(frozen=True, slots=True)
class FrictionRegion:
    """A seeded hinge-friction region in normalized door travel."""

    position_start: float
    position_end: float
    strength: float
    pitch_scale: float

    def __post_init__(self) -> None:
        values = (self.position_start, self.position_end, self.strength, self.pitch_scale)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("friction region values must be finite")
        if not 0.0 <= self.position_start < self.position_end <= 1.0:
            raise ValueError("friction region must have ordered bounds within normalized travel")
        if self.strength < 0.0 or self.pitch_scale <= 0.0:
            raise ValueError("friction strength must be non-negative and pitch scale positive")


@dataclass(frozen=True, slots=True)
class DoorModel:
    """Stable generated physical identity shared by open and close actions."""

    material: str
    seed: int
    panel: ModalBody
    frame: ModalBody
    latch: ModalBody
    hinge_friction: FrictionProfile
    hinge_regions: tuple[FrictionRegion, ...]
    handle_contact: ContactProfile
    latch_contact: ContactProfile
    frame_contact: ContactProfile


def _vary_contact(profile: ContactProfile, rng: np.random.Generator) -> ContactProfile:
    def varied(value: float, spread: float) -> float:
        return float(np.clip(value * float(rng.uniform(1.0 - spread, 1.0 + spread)), 0.0, 1.0))

    return ContactProfile(
        hardness=varied(profile.hardness, 0.09),
        brightness=varied(profile.brightness, 0.12),
        roughness=varied(profile.roughness, 0.16),
        contact_gain=profile.contact_gain * float(rng.uniform(0.88, 1.12)),
    )


def _vary_friction(profile: FrictionProfile, rng: np.random.Generator) -> FrictionProfile:
    shift = float(rng.uniform(0.9, 1.1))
    return replace(
        profile,
        base_gain=profile.base_gain * float(rng.uniform(0.88, 1.12)),
        roughness=float(np.clip(profile.roughness * rng.uniform(0.86, 1.14), 0.0, 1.0)),
        f0_hz=(profile.f0_hz[0] * shift, profile.f0_hz[1] * shift),
        chaos_amount=float(np.clip(profile.chaos_amount * rng.uniform(0.88, 1.12), 0.0, 1.0)),
    )


def _generate_body(
    *,
    bands: tuple[ModeBand, ...],
    size_scale: float,
    damping_scale: float,
    brightness_scale: float,
    seed: int,
    component: str,
    sample_rate: int,
) -> ModalBody:
    return generate_modal_body(
        base_bands=bands,
        size_scale=size_scale,
        damping_scale=damping_scale,
        brightness_scale=brightness_scale,
        rng=door_rng(seed, component),
        sample_rate=sample_rate,
    )


def _hinge_regions(seed: int) -> tuple[FrictionRegion, ...]:
    rng = door_rng(seed, "regions")
    centers = np.sort(rng.uniform(0.13, 0.87, 3))
    regions: list[FrictionRegion] = []
    for center_value in centers:
        center = float(center_value)
        half_width = float(rng.uniform(0.035, 0.075))
        start = max(0.02, center - half_width)
        end = min(0.98, center + half_width)
        regions.append(
            FrictionRegion(
                position_start=start,
                position_end=end,
                strength=float(rng.uniform(0.55, 1.0)),
                pitch_scale=float(rng.uniform(0.82, 1.2)),
            )
        )
    return tuple(regions)


def generate_door_model(*, material: str, seed: int, sample_rate: int) -> DoorModel:
    """Generate a stable semantic door object, not an engineering geometry model."""
    try:
        profile: DoorMaterialProfile = DOOR_MATERIAL_PROFILES[material]
    except KeyError as exc:
        raise ValueError(f"unsupported door material {material!r}; expected wood or metal") from exc
    if seed < 0:
        raise ValueError("seed must be non-negative")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")

    panel_rng = door_rng(seed, "panel-properties")
    panel_size = float(panel_rng.uniform(*profile.panel_size_scale))
    damping_rng = door_rng(seed, "damping")
    panel_damping = float(damping_rng.uniform(*profile.damping_scale))
    brightness = 0.86 if material == "wood" else 1.18
    panel = _generate_body(
        bands=profile.panel_mode_bands,
        size_scale=panel_size,
        damping_scale=panel_damping,
        brightness_scale=brightness,
        seed=seed,
        component="panel",
        sample_rate=sample_rate,
    )
    frame = _generate_body(
        bands=profile.frame_mode_bands,
        size_scale=1.24 if material == "wood" else 1.08,
        damping_scale=panel_damping * (0.76 if material == "wood" else 0.9),
        brightness_scale=brightness * 0.95,
        seed=seed,
        component="frame",
        sample_rate=sample_rate,
    )
    latch = _generate_body(
        bands=profile.latch_mode_bands,
        size_scale=0.62 if material == "wood" else 0.52,
        damping_scale=panel_damping * (0.62 if material == "wood" else 0.8),
        brightness_scale=brightness * 1.08,
        seed=seed,
        component="latch",
        sample_rate=sample_rate,
    )
    return DoorModel(
        material=material,
        seed=seed,
        panel=panel,
        frame=frame,
        latch=latch,
        hinge_friction=_vary_friction(profile.hinge_friction, door_rng(seed, "hinge")),
        hinge_regions=_hinge_regions(seed),
        handle_contact=_vary_contact(profile.handle_contact, door_rng(seed, "handle")),
        latch_contact=_vary_contact(profile.latch_contact, door_rng(seed, "latch-contact")),
        frame_contact=_vary_contact(profile.frame_contact, door_rng(seed, "frame-contact")),
    )


def _mix_at(destination: FloatAudio, source: FloatAudio, start: int) -> None:
    if start < 0:
        source = source[-start:]
        start = 0
    if start >= destination.size or source.size == 0:
        return
    end = min(destination.size, start + source.size)
    destination[start:end] += source[: end - start]


def _limit_peak(signal: FloatAudio, peak: float = 0.95) -> FloatAudio:
    maximum = float(np.max(np.abs(signal))) if signal.size else 0.0
    if maximum > peak:
        signal = signal * np.float32(peak / maximum)
    return np.asarray(signal, dtype=np.float32)


def _region_activity(
    position: FloatAudio, regions: tuple[FrictionRegion, ...]
) -> tuple[FloatAudio, FloatAudio]:
    activity = np.zeros(position.size, dtype=np.float32)
    pitch = np.ones(position.size, dtype=np.float32)
    strongest = np.zeros(position.size, dtype=np.float32)
    for region in regions:
        span = region.position_end - region.position_start
        local = np.clip((position - region.position_start) / np.float32(span), 0.0, 1.0)
        envelope = np.maximum(np.sin(np.float32(math.pi) * local), 0.0) ** np.float32(0.72)
        contribution = envelope * np.float32(region.strength)
        pitch = np.where(contribution > strongest, np.float32(region.pitch_scale), pitch)
        strongest = np.maximum(strongest, contribution)
        activity = np.maximum(activity, contribution)
    return activity, pitch


def render_open(
    *,
    model: DoorModel,
    speed: str,
    creak: float,
    force: float,
    sample_rate: int,
) -> FloatAudio:
    """Compose an opening action by exciting one generated physical door."""
    durations = {"slow": 2.25, "normal": 1.42, "fast": 0.78}
    if speed not in durations:
        raise ValueError("speed must be slow, normal, or fast")
    if not math.isfinite(creak) or not 0.0 <= creak <= 1.0:
        raise ValueError("creak must be finite and between 0 and 1")
    if not math.isfinite(force) or not 0.05 <= force <= 1.0:
        raise ValueError("force must be finite and between 0.05 and 1")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")

    duration = durations[speed]
    motion_rng = door_rng(model.seed, "motion-open")
    acceleration = {"slow": 1.26, "normal": 1.08, "fast": 0.9}[speed]
    motion = eased_motion(
        duration_s=duration,
        sample_rate=sample_rate,
        acceleration_shape=acceleration + float(motion_rng.uniform(-0.04, 0.04)),
        deceleration_shape=1.58 + float(motion_rng.uniform(-0.06, 0.06)),
    )
    size = motion.position.size
    tail_size = round(0.24 * sample_rate)
    result = np.zeros(size + tail_size, dtype=np.float32)

    handle_force = float(np.clip(0.16 + 0.26 * force, 0.05, 1.0))
    handle = impact(
        contact=model.handle_contact,
        force=handle_force,
        duration_s=0.12,
        body=model.latch,
        sample_rate=sample_rate,
        rng=door_rng(model.seed, "handle-event"),
    )
    _mix_at(result, handle, 0)

    latch_rng = door_rng(model.seed, "latch-event")
    latch_delay = round(float(latch_rng.uniform(0.035, 0.082)) * sample_rate)
    latch = impact(
        contact=model.latch_contact,
        force=float(np.clip(0.22 + 0.38 * force, 0.05, 1.0)),
        duration_s=0.18,
        body=model.latch,
        sample_rate=sample_rate,
        rng=latch_rng,
    )
    _mix_at(result, latch, latch_delay)

    breakaway = impact(
        contact=model.handle_contact,
        force=float(np.clip(0.18 + 0.40 * force, 0.05, 1.0)),
        duration_s=0.15,
        body=model.panel,
        sample_rate=sample_rate,
        rng=door_rng(model.seed, "breakaway"),
    )
    _mix_at(result, breakaway, round(0.12 * sample_rate))

    activity, pitch_scale = _region_activity(motion.position, model.hinge_regions)
    hinge_profile = replace(
        model.hinge_friction,
        base_gain=model.hinge_friction.base_gain * (0.12 + 0.88 * creak),
        roughness=model.hinge_friction.roughness * (0.18 + 0.82 * creak),
        stick_strength=model.hinge_friction.stick_strength * creak,
        slip_strength=model.hinge_friction.slip_strength * (0.25 + 0.75 * creak),
    )
    hinge = friction(
        motion=motion,
        profile=hinge_profile,
        body=model.panel,
        sample_rate=sample_rate,
        rng=door_rng(model.seed, "hinge-event"),
        activity=activity,
        pitch_scale=pitch_scale,
    )
    _mix_at(result, hinge, 0)

    terminal = impact(
        contact=model.frame_contact,
        force=float(np.clip(0.10 + 0.17 * force, 0.05, 0.5)),
        duration_s=0.19,
        body=model.panel,
        sample_rate=sample_rate,
        rng=door_rng(model.seed, "terminal-stop"),
    )
    _mix_at(result, terminal, size - round(0.035 * sample_rate))
    return _limit_peak(result)


def render_close(
    *,
    model: DoorModel,
    speed: str,
    creak: float,
    force: float,
    sample_rate: int,
) -> FloatAudio:
    """Compose closing motion, frame collision, latch catch, and resonant decay."""
    durations = {"slow": 0.82, "normal": 0.46, "fast": 0.23}
    if speed not in durations:
        raise ValueError("speed must be slow, normal, or fast")
    if not math.isfinite(creak) or not 0.0 <= creak <= 1.0:
        raise ValueError("creak must be finite and between 0 and 1")
    if not math.isfinite(force) or not 0.05 <= force <= 1.0:
        raise ValueError("force must be finite and between 0.05 and 1")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")

    motion_rng = door_rng(model.seed, "motion-close")
    motion = eased_motion(
        duration_s=durations[speed],
        sample_rate=sample_rate,
        acceleration_shape=0.9 + float(motion_rng.uniform(-0.06, 0.06)),
        deceleration_shape=1.22 + float(motion_rng.uniform(-0.08, 0.08)),
    )
    size = motion.position.size
    result = np.zeros(size + round(0.55 * sample_rate), dtype=np.float32)

    activity, pitch_scale = _region_activity(motion.position, model.hinge_regions)
    hinge_profile = replace(
        model.hinge_friction,
        base_gain=model.hinge_friction.base_gain * (0.08 + 0.45 * creak),
        roughness=model.hinge_friction.roughness * (0.16 + 0.55 * creak),
        stick_strength=model.hinge_friction.stick_strength * creak * 0.58,
        slip_strength=model.hinge_friction.slip_strength * (0.2 + 0.55 * creak),
    )
    movement_friction = friction(
        motion=motion,
        profile=hinge_profile,
        body=model.panel,
        sample_rate=sample_rate,
        rng=door_rng(model.seed, "hinge-event", 1),
        activity=activity,
        pitch_scale=pitch_scale,
    )
    _mix_at(result, movement_friction, 0)

    impact_start = size - round(0.006 * sample_rate)
    frame_hit = impact(
        contact=model.frame_contact,
        force=force,
        duration_s=0.36,
        body=(model.panel, model.frame),
        sample_rate=sample_rate,
        rng=door_rng(model.seed, "frame-impact"),
    )
    _mix_at(result, frame_hit, impact_start)
    event_end = impact_start + frame_hit.size

    latch_rng = door_rng(model.seed, "latch-catch")
    latch_delay = round(float(latch_rng.uniform(0.03, 0.14)) * sample_rate)
    latch = impact(
        contact=model.latch_contact,
        force=float(np.clip(0.24 + 0.55 * force, 0.05, 1.0)),
        duration_s=0.18,
        body=model.latch,
        sample_rate=sample_rate,
        rng=latch_rng,
    )
    latch_start = size + latch_delay
    _mix_at(result, latch, latch_start)
    event_end = max(event_end, latch_start + latch.size)

    chatter_rng = door_rng(model.seed, "chatter")
    chatter_probability = 0.03 + 0.34 * force
    if float(chatter_rng.random()) < chatter_probability:
        chatter_count = 1 + int(chatter_rng.random() < 0.24 * force)
        for event_index in range(chatter_count):
            event_rng = door_rng(model.seed, "chatter", event_index + 1)
            delay = round(float(event_rng.uniform(0.035, 0.095)) * sample_rate)
            chatter = impact(
                contact=model.latch_contact,
                force=float(np.clip(0.14 + 0.28 * force, 0.05, 0.5)),
                duration_s=0.11,
                body=model.latch,
                sample_rate=sample_rate,
                rng=event_rng,
            )
            chatter_start = latch_start + delay
            _mix_at(result, chatter, chatter_start)
            event_end = max(event_end, chatter_start + chatter.size)

    return _limit_peak(result[:event_end])

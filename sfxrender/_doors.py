"""Generated door identity and action-independent physical properties."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

from ._door_profiles import DOOR_MATERIAL_PROFILES, DoorMaterialProfile
from ._dsp import one_pole_highpass, one_pole_lowpass
from ._physics.contact import ImpactContact, impact_force
from ._physics.friction import (
    LuGrePreset,
    lugre_friction,
    make_roughness_profile,
    roughness_velocity,
)
from ._physics.geometry import BoundaryCondition, RectangularPlate, rectangular_plate_modes
from ._physics.materials import OAK_EFFECTIVE, PAINTED_STEEL, STEEL_SHEET, MechanicalMaterial
from ._physics.models import ContactProfile, FrictionProfile, ModalBody, ModeBand, MotionCurve
from ._physics.modes import ModeSet
from ._physics.motion import minimum_jerk_motion
from ._physics.physical_impacts import render_physical_impact
from ._physics.resonator import ModalResonatorBank
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
    mechanical_material: MechanicalMaterial
    panel_modes: ModeSet
    frame_modes: ModeSet
    latch_modes: ModeSet
    hinge_mode_sets: tuple[ModeSet, ModeSet, ModeSet]
    width_m: float
    height_m: float
    thickness_m: float
    mass_kg: float
    inertia_kg_m2: float
    hinge_preload_n: float


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


def _physical_door_properties(
    material: str, sample_rate: int
) -> tuple[
    MechanicalMaterial,
    ModeSet,
    ModeSet,
    ModeSet,
    tuple[ModeSet, ModeSet, ModeSet],
    float,
    float,
    float,
    float,
    float,
    float,
]:
    if material == "wood":
        mechanical_material = OAK_EFFECTIVE
        width_m, height_m, thickness_m, mass_kg, preload_n = 0.86, 2.03, 0.040, 27.0, 55.0
    else:
        mechanical_material = PAINTED_STEEL
        width_m, height_m, thickness_m, mass_kg, preload_n = 0.90, 2.05, 0.0025, 36.0, 90.0
    plate = RectangularPlate(width_m, height_m, thickness_m, BoundaryCondition.DOOR_HINGED)
    panel_modes = rectangular_plate_modes(
        plate,
        mechanical_material,
        sample_rate=sample_rate,
        max_modes=32,
        contact_x=0.72,
        contact_y=0.54,
    )
    frame_modes = rectangular_plate_modes(
        RectangularPlate(0.11, height_m, 0.018, BoundaryCondition.WALL_PANEL),
        mechanical_material,
        sample_rate=sample_rate,
        max_modes=20,
        contact_x=0.5,
        contact_y=0.52,
    )
    latch_modes = rectangular_plate_modes(
        RectangularPlate(0.055, 0.095, 0.002, BoundaryCondition.CLAMPED),
        STEEL_SHEET,
        sample_rate=sample_rate,
        max_modes=16,
        contact_x=0.64,
        contact_y=0.57,
    )

    def hinge_modes_at(height_position: float) -> ModeSet:
        return rectangular_plate_modes(
            plate,
            mechanical_material,
            sample_rate=sample_rate,
            max_modes=24,
            contact_x=0.055,
            contact_y=height_position,
        )

    hinge_modes = (
        hinge_modes_at(0.10),
        hinge_modes_at(0.50),
        hinge_modes_at(0.90),
    )
    inertia = mass_kg * width_m**2 / 3.0
    return (
        mechanical_material,
        panel_modes,
        frame_modes,
        latch_modes,
        hinge_modes,
        width_m,
        height_m,
        thickness_m,
        mass_kg,
        inertia,
        preload_n,
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
    (
        mechanical_material,
        panel_modes,
        frame_modes,
        latch_modes,
        hinge_mode_sets,
        width_m,
        height_m,
        thickness_m,
        mass_kg,
        inertia_kg_m2,
        hinge_preload_n,
    ) = _physical_door_properties(material, sample_rate)
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
        mechanical_material=mechanical_material,
        panel_modes=panel_modes,
        frame_modes=frame_modes,
        latch_modes=latch_modes,
        hinge_mode_sets=hinge_mode_sets,
        width_m=width_m,
        height_m=height_m,
        thickness_m=thickness_m,
        mass_kg=mass_kg,
        inertia_kg_m2=inertia_kg_m2,
        hinge_preload_n=hinge_preload_n,
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
    magnitude = np.abs(signal)
    knee = peak * 0.88
    compressed = knee + (peak - knee) * (
        1.0 - np.exp(-np.maximum(magnitude - knee, 0.0) / (peak - knee))
    )
    limited = np.minimum(np.where(magnitude <= knee, magnitude, compressed), np.float32(peak))
    return np.asarray(np.sign(signal) * limited, dtype=np.float32)


def _door_tail_seconds(model: DoorModel) -> float:
    modes = (model.panel_modes, model.frame_modes, model.latch_modes)
    decay = max(mode.decay_s for mode_set in modes for mode in mode_set.modes)
    return min(0.9, max(0.24, decay * 4.5))


def _mechanical_impact(
    model: DoorModel,
    contact: ContactProfile,
    modes: ModeSet,
    force: float,
    sample_rate: int,
    rng: np.random.Generator,
) -> FloatAudio:
    material = model.mechanical_material
    effective_mass = 0.012 + 0.045 * (1.0 - contact.hardness)
    stiffness = 2.5e5 * 10.0 ** (1.35 * contact.hardness)
    velocity = 0.12 + 1.75 * force
    trace = impact_force(
        contact=ImpactContact(
            effective_mass_kg=effective_mass,
            stiffness=stiffness,
            restitution=material.restitution,
        ),
        velocity_m_s=velocity,
        sample_rate=sample_rate,
    )
    rendered = render_physical_impact(
        trace,
        modes,
        sample_rate=sample_rate,
        rng=rng,
        microscopic_gain=(0.32 + 0.42 * contact.brightness)
        * (1.8 if model.material == "metal" else 0.75),
        output_gain=22.0 + 20.0 * contact.contact_gain,
    )
    return np.asarray(rendered * np.float32(0.4 + force), dtype=np.float32)


def _door_radiation(signal: FloatAudio, model: DoorModel, sample_rate: int) -> FloatAudio:
    if model.material == "wood":
        filtered = one_pole_lowpass(signal, sample_rate, 700.0)
        return one_pole_lowpass(filtered, sample_rate, 700.0)
    return one_pole_highpass(signal, sample_rate, 700.0)


def _hinge_friction_audio(
    model: DoorModel,
    motion: MotionCurve,
    creak: float,
    sample_rate: int,
    *,
    closing: bool,
    tail_s: float,
) -> FloatAudio:
    position = np.asarray(motion.position, dtype=np.float32)
    direction = -1.0 if closing else 1.0
    angle_rad = 1.35
    angular_velocity = np.asarray(motion.velocity, dtype=np.float32) * np.float32(
        angle_rad * direction
    )
    angular_acceleration = np.asarray(motion.acceleration, dtype=np.float32) * np.float32(
        angle_rad * direction
    )
    tangent_velocity = angular_velocity * np.float32(0.022)
    reaction = np.abs(model.inertia_kg_m2 * angular_acceleration) / (model.width_m * 3.0)
    normal_load = np.float32(model.hinge_preload_n / 3.0) + reaction
    path = position if not closing else np.float32(1.0) - position
    travel_m = 0.030
    output_size = position.size + round(tail_s * sample_rate)
    output = np.zeros(output_size, dtype=np.float32)
    friction_factor = 0.18 + 0.82 * creak
    preset = LuGrePreset(
        static_coefficient=model.mechanical_material.friction_static * friction_factor,
        dynamic_coefficient=model.mechanical_material.friction_dynamic * friction_factor,
        stribeck_velocity_m_s=model.mechanical_material.stribeck_velocity_m_s,
        bristle_stiffness_n_m=72_000.0 * (0.72 + 0.55 * creak),
        bristle_damping_n_s_m=0.08,
        viscous_coefficient_n_s_m=model.mechanical_material.viscous_friction,
    )
    tail_samples = output_size - position.size
    for hinge_index, modes in enumerate(model.hinge_mode_sets):
        rng = door_rng(model.seed, "hinge-event", hinge_index)
        roughness = make_roughness_profile(
            roughness_rms_m=model.mechanical_material.roughness_rms_m * (0.08 + 0.92 * creak),
            correlation_length_m=model.mechanical_material.roughness_correlation_m,
            seed=int(rng.integers(0, 2**32, dtype=np.uint32)),
            length_m=0.15,
        )
        state = lugre_friction(
            np.asarray(tangent_velocity, dtype=np.float32),
            np.asarray(normal_load, dtype=np.float32),
            preset,
            sample_rate,
        )
        contact_path = np.asarray(
            path * np.float32(travel_m) + np.float32(hinge_index * 0.018),
            dtype=np.float32,
        )
        surface_velocity = roughness_velocity(roughness, contact_path, sample_rate)
        if state.power_w.size and float(np.max(state.power_w)) > 1e-12:
            activity = np.sqrt(state.power_w / np.float32(np.max(state.power_w)))
        else:
            activity = np.zeros(position.size, dtype=np.float32)
        texture = surface_velocity * np.float32(2_600.0 * (0.2 + 0.8 * creak)) * activity
        release = np.sqrt(state.release_energy_j * np.float32(sample_rate)) * np.float32(
            0.025 * creak
        )
        excitation = np.asarray(texture + release, dtype=np.float32)
        bank = ModalResonatorBank(modes, sample_rate)
        padded = np.pad(excitation, (0, tail_samples))
        output += bank.process(np.asarray(padded, dtype=np.float32)) * np.float32(850.0)
    return output


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
    motion = minimum_jerk_motion(duration_s=duration, sample_rate=sample_rate)
    size = motion.position.size
    tail_size = round(_door_tail_seconds(model) * sample_rate)
    result = np.zeros(size + tail_size, dtype=np.float32)
    door_modes = ModeSet(model.latch_modes.modes + model.panel_modes.modes)
    frame_modes = ModeSet(model.frame_modes.modes + model.panel_modes.modes)

    handle_force = float(np.clip(0.16 + 0.26 * force, 0.05, 1.0))
    handle = _mechanical_impact(
        model,
        model.handle_contact,
        door_modes,
        handle_force,
        sample_rate,
        door_rng(model.seed, "handle-event"),
    )
    _mix_at(result, handle, 0)

    latch_rng = door_rng(model.seed, "latch-event")
    latch_delay = round(float(latch_rng.uniform(0.035, 0.082)) * sample_rate)
    latch = _mechanical_impact(
        model,
        model.latch_contact,
        door_modes,
        float(np.clip(0.22 + 0.38 * force, 0.05, 1.0)),
        sample_rate,
        latch_rng,
    )
    _mix_at(result, latch, latch_delay)

    breakaway = _mechanical_impact(
        model,
        model.handle_contact,
        model.panel_modes,
        float(np.clip(0.18 + 0.40 * force, 0.05, 1.0)),
        sample_rate,
        door_rng(model.seed, "breakaway"),
    )
    _mix_at(result, breakaway, round(0.12 * sample_rate))

    hinge = _hinge_friction_audio(
        model,
        motion,
        creak,
        sample_rate,
        closing=False,
        tail_s=_door_tail_seconds(model),
    )
    _mix_at(result, hinge, 0)

    terminal = _mechanical_impact(
        model,
        model.frame_contact,
        frame_modes,
        float(np.clip(0.10 + 0.17 * force, 0.05, 0.5)),
        sample_rate,
        door_rng(model.seed, "terminal-stop"),
    )
    _mix_at(result, terminal, size - round(0.035 * sample_rate))
    return _limit_peak(_door_radiation(result, model, sample_rate))


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

    motion = minimum_jerk_motion(duration_s=durations[speed], sample_rate=sample_rate)
    size = motion.position.size
    tail_s = _door_tail_seconds(model)
    result = np.zeros(size + round(max(0.55, tail_s) * sample_rate), dtype=np.float32)
    door_modes = ModeSet(model.latch_modes.modes + model.panel_modes.modes)
    frame_modes = ModeSet(model.frame_modes.modes + model.panel_modes.modes)

    movement_friction = _hinge_friction_audio(
        model,
        motion,
        creak,
        sample_rate,
        closing=True,
        tail_s=tail_s,
    )
    _mix_at(result, movement_friction, 0)

    impact_start = size - round(0.006 * sample_rate)
    frame_hit = _mechanical_impact(
        model,
        model.frame_contact,
        frame_modes,
        force,
        sample_rate,
        door_rng(model.seed, "frame-impact"),
    )
    _mix_at(result, frame_hit, impact_start)
    event_end = impact_start + frame_hit.size

    latch_rng = door_rng(model.seed, "latch-catch")
    latch_delay = round(float(latch_rng.uniform(0.03, 0.14)) * sample_rate)
    latch = _mechanical_impact(
        model,
        model.latch_contact,
        door_modes,
        float(np.clip(0.24 + 0.55 * force, 0.05, 1.0)),
        sample_rate,
        latch_rng,
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
            chatter = _mechanical_impact(
                model,
                model.latch_contact,
                door_modes,
                float(np.clip(0.14 + 0.28 * force, 0.05, 0.5)),
                sample_rate,
                event_rng,
            )
            chatter_start = latch_start + delay
            _mix_at(result, chatter, chatter_start)
            event_end = max(event_end, chatter_start + chatter.size)

    return _limit_peak(_door_radiation(result[:event_end], model, sample_rate))

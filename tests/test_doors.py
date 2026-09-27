"""Tests for generated, seed-stable physical door identities."""

from __future__ import annotations

import numpy as np
import pytest

from sfxrender import SFXRenderer
from sfxrender._doors import door_rng, generate_door_model


def test_door_model_identity_is_deterministic_and_seed_specific() -> None:
    wood = generate_door_model(material="wood", seed=42, sample_rate=24_000)
    repeated = generate_door_model(material="wood", seed=42, sample_rate=24_000)
    other_seed = generate_door_model(material="wood", seed=43, sample_rate=24_000)

    assert wood == repeated
    assert wood != other_seed
    assert wood.panel != other_seed.panel or wood.hinge_regions != other_seed.hinge_regions
    assert len(wood.hinge_regions) == 3
    assert all(
        0.0 <= region.position_start < region.position_end <= 1.0 for region in wood.hinge_regions
    )
    assert (
        tuple(sorted(wood.hinge_regions, key=lambda region: region.position_start))
        == wood.hinge_regions
    )
    assert (
        wood.handle_contact != other_seed.handle_contact
        or wood.latch_contact != other_seed.latch_contact
    )


def test_door_substreams_use_stable_independent_component_ids() -> None:
    panel = door_rng(51, "panel").normal(size=8)
    repeated_panel = door_rng(51, "panel").normal(size=8)
    latch = door_rng(51, "latch").normal(size=8)
    np.testing.assert_array_equal(panel, repeated_panel)
    assert not np.array_equal(panel, latch)
    assert not np.array_equal(
        door_rng(51, "panel").normal(size=8), door_rng(51, "panel", 1).normal(size=8)
    )


def test_generated_door_modes_are_finite_and_nyquist_safe_at_supported_rates() -> None:
    for sample_rate in (8_000, 12_000, 24_000):
        for material in ("wood", "metal"):
            model = generate_door_model(material=material, seed=88, sample_rate=sample_rate)
            for body in (model.panel, model.frame, model.latch):
                assert all(
                    np.isfinite((mode.frequency_hz, mode.decay_s, mode.gain)).all()
                    and 0.0 < mode.frequency_hz < sample_rate * 0.45
                    and mode.decay_s > 0.0
                    and mode.gain >= 0.0
                    for mode in body.modes
                )


def test_metal_door_panel_has_more_high_frequency_modal_gain_in_aggregate() -> None:
    wood_ratios: list[float] = []
    metal_ratios: list[float] = []
    for seed in (3, 17, 42, 107, 211):
        wood = generate_door_model(material="wood", seed=seed, sample_rate=24_000).panel
        metal = generate_door_model(material="metal", seed=seed, sample_rate=24_000).panel
        for body, ratios in ((wood, wood_ratios), (metal, metal_ratios)):
            total = sum(mode.gain for mode in body.modes)
            high = sum(mode.gain for mode in body.modes if mode.frequency_hz >= 1_500.0)
            ratios.append(high / total if total else 0.0)
    assert float(np.median(metal_ratios)) > float(np.median(wood_ratios))


def test_unknown_door_material_is_rejected() -> None:
    with pytest.raises(ValueError, match="wood or metal"):
        generate_door_model(material="glass", seed=1, sample_rate=24_000)


def _rms(samples: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))


def test_open_is_seeded_and_speed_controls_duration() -> None:
    renderer = SFXRenderer(sample_rate=12_000)
    base = "sfx:door.open?material=wood&creak=0.7&seed=103"
    normal = renderer.render_uri(base + "&speed=normal")
    repeated = renderer.render_uri(base + "&speed=normal")
    slow = renderer.render_uri(base + "&speed=slow")
    fast = renderer.render_uri(base + "&speed=fast")
    changed = renderer.render_uri("sfx:door.open?material=wood&creak=0.7&speed=normal&seed=104")

    np.testing.assert_array_equal(normal.samples, repeated.samples)
    assert not np.array_equal(normal.samples, changed.samples)
    assert slow.duration > normal.duration > fast.duration
    assert normal.samples.dtype == np.float32
    assert np.all(np.isfinite(normal.samples))
    assert float(np.max(np.abs(normal.samples))) <= 0.95


def test_open_creak_changes_hinge_energy_but_zero_keeps_mechanical_sources() -> None:
    renderer = SFXRenderer(sample_rate=12_000)
    quiet = renderer.render_uri("sfx:door.open?material=wood&speed=normal&creak=0&seed=72").samples
    creaky = renderer.render_uri(
        "sfx:door.open?material=wood&speed=normal&creak=0.9&seed=72"
    ).samples
    movement = slice(int(0.28 * 12_000), int(1.25 * 12_000))

    assert _rms(quiet) > 0.0
    assert _rms(quiet[movement]) > 0.0
    assert _rms(creaky[movement]) > _rms(quiet[movement]) * 1.15


def test_open_force_changes_hardware_excitation() -> None:
    renderer = SFXRenderer(sample_rate=12_000)
    light = renderer.render_uri(
        "sfx:door.open?material=wood&speed=normal&force=0.15&seed=95"
    ).samples
    firm = renderer.render_uri(
        "sfx:door.open?material=wood&speed=normal&force=0.95&seed=95"
    ).samples

    assert _rms(firm[: int(0.28 * 12_000)]) > _rms(light[: int(0.28 * 12_000)]) * 1.3


def test_close_is_seeded_registered_and_speed_controls_duration() -> None:
    renderer = SFXRenderer(sample_rate=12_000)
    base = "sfx:door.close?material=wood&creak=0.2&force=0.72&seed=205"
    normal = renderer.render_uri(base + "&speed=normal")
    repeated = renderer.render_uri(base + "&speed=normal")
    slow = renderer.render_uri(base + "&speed=slow")
    fast = renderer.render_uri(base + "&speed=fast")
    changed = renderer.render_uri(
        "sfx:door.close?material=wood&creak=0.2&force=0.72&speed=normal&seed=206"
    )

    assert "door.close" in renderer.effects()
    np.testing.assert_array_equal(normal.samples, repeated.samples)
    assert not np.array_equal(normal.samples, changed.samples)
    assert slow.duration > normal.duration > fast.duration
    assert normal.samples.dtype == np.float32
    assert np.all(np.isfinite(normal.samples))
    assert float(np.max(np.abs(normal.samples))) <= 0.95


def test_close_terminal_collision_dominates_and_force_changes_impact() -> None:
    renderer = SFXRenderer(sample_rate=12_000)
    uri = "sfx:door.close?material=wood&speed=normal&creak=0.2&seed=61&force="
    light = renderer.render_uri(uri + "0.12").samples
    strong = renderer.render_uri(uri + "1.0").samples
    before = slice(int(0.06 * 12_000), int(0.38 * 12_000))
    collision = slice(int(0.455 * 12_000), int(0.60 * 12_000))

    assert _rms(strong[collision]) > _rms(light[collision]) * 1.3
    assert _rms(strong[collision]) > _rms(strong[before]) * 1.2


def _band_fraction(samples: np.ndarray, sample_rate: int, low_hz: float, high_hz: float) -> float:
    windowed = samples * np.hanning(samples.size)
    power = np.square(np.abs(np.fft.rfft(windowed)))
    frequencies = np.fft.rfftfreq(samples.size, 1.0 / sample_rate)
    mask = (frequencies >= low_hz) & (frequencies < high_hz)
    total = float(power.sum())
    return float(power[mask].sum() / total) if total > 0.0 else 0.0


def test_open_motion_is_intermittent_and_wood_avoids_high_band_hiss() -> None:
    sample_rate = 24_000
    sound = (
        SFXRenderer(sample_rate=sample_rate)
        .render_uri("sfx:door.open?material=wood&speed=slow&creak=0.75&seed=31")
        .samples
    )
    movement = sound[int(0.30 * sample_rate) : int(2.15 * sample_rate)]
    frame_size = round(0.05 * sample_rate)
    frames = movement[: movement.size // frame_size * frame_size].reshape(-1, frame_size)
    frame_rms = np.sqrt(np.mean(np.square(frames.astype(np.float64)), axis=1))
    high_noise_fraction = _band_fraction(movement, sample_rate, 5_000.0, 8_000.0)

    assert float(np.percentile(frame_rms, 90)) > float(np.percentile(frame_rms, 25)) * 4.0
    assert high_noise_fraction < 0.1


def test_same_seed_open_and_close_wrappers_share_door_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sfxrender import procedural
    from sfxrender.types import RenderContext, SfxSpec

    captured: list[object] = []
    silence = np.zeros(1, dtype=np.float32)

    def capture_open(*, model, speed, creak, force, sample_rate):
        captured.append(model)
        return silence

    def capture_close(*, model, speed, creak, force, sample_rate):
        captured.append(model)
        return silence

    monkeypatch.setattr(procedural, "render_open", capture_open)
    monkeypatch.setattr(procedural, "render_close", capture_close)
    context = RenderContext(sample_rate=8_000)
    procedural.door_open(
        SfxSpec("door.open", {"material": "wood", "seed": "501", "force": "0.05"}),
        context,
    )
    procedural.door_close(
        SfxSpec("door.close", {"material": "wood", "seed": "501", "force": "0.95"}),
        context,
    )

    assert captured[0] == captured[1]


def test_metal_close_tail_has_more_high_frequency_energy_across_seeds() -> None:
    renderer = SFXRenderer(sample_rate=24_000)
    wood_fractions: list[float] = []
    metal_fractions: list[float] = []
    for seed in (3, 17, 42, 107, 211):
        for material, collected in (("wood", wood_fractions), ("metal", metal_fractions)):
            samples = renderer.render_uri(
                f"sfx:door.close?material={material}&speed=normal&force=0.7&seed={seed}"
            ).samples
            tail = samples[int(0.40 * 24_000) :]
            collected.append(_band_fraction(tail, 24_000, 1_200.0, 8_000.0))

    assert float(np.median(metal_fractions)) > float(np.median(wood_fractions)) * 1.5

"""Tests for shared physical footstep contacts and granular work bounds."""

from __future__ import annotations

import numpy as np

from sfxrender._foley_profiles import AGGREGATE_SURFACE_PROFILES, FOOTWEAR_PROFILES
from sfxrender._footsteps import (
    _floor_modes,
    _mechanical_friction_audio,
    _sample_particle_events,
    solid_footstep,
)
from sfxrender._physics.presets import FLOOR_OBJECTS, FOOTWEAR_CONTACTS


def test_floor_and_footwear_presets_provide_distinct_physical_objects() -> None:
    wood = _floor_modes("wood", 16_000, 0.2, 0.4)
    stone = _floor_modes("stone", 16_000, 0.2, 0.4)
    carpet = _floor_modes("carpet", 16_000, 0.2, 0.4)
    assert FLOOR_OBJECTS["wood"].geometry.width_m > FLOOR_OBJECTS["stone"].geometry.width_m
    assert wood.modes and stone.modes and carpet.modes
    assert wood.modes[0].frequency_hz != stone.modes[0].frequency_hz
    assert max(mode.decay_s for mode in carpet.modes) < max(mode.decay_s for mode in wood.modes)
    assert (
        FOOTWEAR_CONTACTS["heels"].normal_stiffness_n_m
        > FOOTWEAR_CONTACTS["boots"].normal_stiffness_n_m
    )
    assert (
        FOOTWEAR_CONTACTS["boots"].effective_mass_kg > FOOTWEAR_CONTACTS["shoes"].effective_mass_kg
    )


def test_loaded_slip_excitation_increases_with_ground_load() -> None:
    size = 3_200
    low_grf = np.full(size, 0.2, dtype=np.float32)
    high_grf = np.full(size, 0.8, dtype=np.float32)
    modes = _floor_modes("wood", 16_000, 0.62, 0.56)
    common_footwear = FOOTWEAR_PROFILES["boots"]
    contact = FOOTWEAR_CONTACTS["boots"]
    low = _mechanical_friction_audio(
        grf=low_grf,
        footwear=common_footwear,
        footwear_contact=contact,
        surface="wood",
        modes=modes,
        rng=np.random.default_rng(32),
        sample_rate=16_000,
    )
    high = _mechanical_friction_audio(
        grf=high_grf,
        footwear=common_footwear,
        footwear_contact=contact,
        surface="wood",
        modes=modes,
        rng=np.random.default_rng(32),
        sample_rate=16_000,
    )
    assert np.all(np.isfinite(high))
    assert float(np.sqrt(np.mean(high**2))) > float(np.sqrt(np.mean(low**2))) * 1.5


def test_particle_collision_energy_is_bounded_by_per_block_slip_work() -> None:
    sample_rate = 16_000
    grf = np.concatenate(
        (np.full(800, 0.85, dtype=np.float32), np.full(800, 0.08, dtype=np.float32))
    )
    events = _sample_particle_events(
        grf=grf,
        surface=AGGREGATE_SURFACE_PROFILES["gravel"],
        rng=np.random.default_rng(91),
        sample_rate=sample_rate,
    )
    block_size = round(0.002 * sample_rate)
    per_block: dict[int, list[float]] = {}
    budgets: dict[int, float] = {}
    for event in events:
        block = event.start_sample // block_size
        per_block.setdefault(block, []).append(event.energy)
        budgets[block] = event.work_budget_j
    assert per_block
    assert sum(event.start_sample < 800 for event in events) > sum(
        event.start_sample >= 800 for event in events
    )
    for block, energies in per_block.items():
        assert sum(energies) <= budgets[block] + 1e-12
        assert all(
            AGGREGATE_SURFACE_PROFILES["gravel"].minimum_energy
            <= energy
            <= AGGREGATE_SURFACE_PROFILES["gravel"].maximum_energy
            for energy in energies
        )


def test_solid_footstep_reuses_physics_deterministically() -> None:
    def render(seed: int) -> np.ndarray:
        return solid_footstep(
            sample_rate=16_000,
            surface="wood",
            footwear="boots",
            force=0.58,
            rng=np.random.default_rng(seed),
        )

    first = render(23)
    repeated = render(23)
    changed = render(24)
    np.testing.assert_array_equal(first, repeated)
    assert not np.array_equal(first, changed)
    assert np.all(np.isfinite(first))
    assert first.dtype == np.float32

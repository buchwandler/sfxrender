"""Invariants for geometry-derived modes and persistent resonators."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from sfxrender._physics.geometry import (
    RectangularPlate,
    plate_mode_shape,
    rectangular_plate_modes,
)
from sfxrender._physics.materials import OAK_EFFECTIVE, MechanicalMaterial
from sfxrender._physics.modes import (
    Mode,
    ModeSet,
    decay_from_q,
    filter_audio_modes,
    q_from_decay,
)
from sfxrender._physics.radiation import radiation_efficiency
from sfxrender._physics.resonator import ModalResonatorBank, modal_response
from sfxrender._physics.rng import RandomStream, event_rng

_PLATE = RectangularPlate(0.72, 0.43, 0.012)


def _modes(plate: RectangularPlate = _PLATE, material: MechanicalMaterial = OAK_EFFECTIVE):
    return rectangular_plate_modes(
        plate, material, sample_rate=24_000, max_frequency_hz=9_000, max_modes=24
    )


def test_plate_scaling_follows_thin_plate_physics() -> None:
    baseline = _modes()
    doubled_size = _modes(replace(_PLATE, width_m=_PLATE.width_m * 2, height_m=_PLATE.height_m * 2))
    doubled_thickness = _modes(replace(_PLATE, thickness_m=_PLATE.thickness_m * 2))
    fourfold_modulus = _modes(
        material=replace(OAK_EFFECTIVE, young_modulus_pa=OAK_EFFECTIVE.young_modulus_pa * 4)
    )
    fourfold_density = _modes(
        material=replace(OAK_EFFECTIVE, density_kg_m3=OAK_EFFECTIVE.density_kg_m3 * 4)
    )
    assert baseline.modes
    assert doubled_size.modes[0].frequency_hz == pytest.approx(baseline.modes[0].frequency_hz / 4)
    assert doubled_thickness.modes[0].frequency_hz == pytest.approx(
        baseline.modes[0].frequency_hz * 2
    )
    assert fourfold_modulus.modes[0].frequency_hz == pytest.approx(
        baseline.modes[0].frequency_hz * 2
    )
    assert fourfold_density.modes[0].frequency_hz == pytest.approx(
        baseline.modes[0].frequency_hz / 2
    )


def test_strike_position_suppresses_modes_near_a_nodal_line() -> None:
    assert abs(plate_mode_shape(2, 1, 0.5, 0.37)) < 1e-12
    centered = rectangular_plate_modes(
        _PLATE, OAK_EFFECTIVE, sample_rate=24_000, max_modes=24, contact_x=0.5, contact_y=0.43
    )
    off_center = rectangular_plate_modes(
        _PLATE, OAK_EFFECTIVE, sample_rate=24_000, max_modes=24, contact_x=0.31, contact_y=0.43
    )
    centered_by_frequency = {mode.frequency_hz: mode.input_gain for mode in centered.modes}
    off_center_by_frequency = {mode.frequency_hz: mode.input_gain for mode in off_center.modes}
    affected = [
        frequency
        for frequency, gain in centered_by_frequency.items()
        if frequency in off_center_by_frequency and abs(gain) < 1e-12
    ]
    assert affected
    assert any(abs(off_center_by_frequency[frequency]) > 0.1 for frequency in affected)


def test_plate_modes_obey_nyquist_limit_and_geometry_validation() -> None:
    modes = rectangular_plate_modes(
        _PLATE, OAK_EFFECTIVE, sample_rate=8_000, max_frequency_hz=10_000, max_modes=100
    )
    assert modes.modes
    assert all(mode.frequency_hz < 8_000 * 0.45 for mode in modes.modes)
    with pytest.raises(ValueError, match="width_m"):
        RectangularPlate(0.0, 1.0, 0.01)


def test_modal_resonator_frequency_decay_and_float32_output() -> None:
    mode_set = ModeSet((Mode(440.0, 0.12, input_gain=1.0, radiation_gain=1.0),))
    excitation = np.zeros(12_000, dtype=np.float32)
    excitation[0] = 1.0
    first = modal_response(excitation, mode_set, 24_000)
    repeated = modal_response(excitation, mode_set, 24_000)
    np.testing.assert_array_equal(first, repeated)
    assert first.dtype == np.float32
    assert np.all(np.isfinite(first))
    peak_hz = np.fft.rfftfreq(first.size, 1 / 24_000)[np.argmax(np.abs(np.fft.rfft(first)))]
    assert peak_hz == pytest.approx(440.0, abs=2.0)
    assert abs(float(first[2_400])) > 0.0
    ratio = abs(float(first[4_800])) / abs(float(first[2_400]))
    assert ratio == pytest.approx(np.exp(-0.1 / 0.12), rel=0.03)


def test_modal_state_persists_across_blocks_and_can_reset() -> None:
    bank = ModalResonatorBank(ModeSet((Mode(350.0, 0.2),)), 12_000)
    impulse = np.zeros(1_000, dtype=np.float32)
    impulse[0] = 1.0
    first = bank.process(impulse)
    second = bank.process(np.zeros(1_000, dtype=np.float32))
    modes = ModeSet((Mode(350.0, 0.2),))
    together = modal_response(np.pad(impulse, (0, 1_000)), modes, 12_000)
    np.testing.assert_allclose(np.concatenate((first, second)), together, rtol=1e-6, atol=1e-7)
    bank.reset()
    assert np.max(np.abs(bank.process(np.zeros(200, dtype=np.float32)))) == 0.0


def test_resonator_is_stable_for_overdamped_and_filters_above_nyquist_modes() -> None:
    modes = ModeSet((Mode(400.0, 1e-5), Mode(5_000.0, 0.2)))
    filtered = filter_audio_modes(modes, sample_rate=8_000, nyquist_fraction=0.46)
    assert len(filtered.modes) == 1
    bank = ModalResonatorBank(modes, 8_000)
    result = bank.process(np.ones(4_000, dtype=np.float32))
    assert result.dtype == np.float32
    assert np.all(np.isfinite(result))
    assert np.max(np.abs(result)) < 1.0


def test_damping_conversions_and_material_validation() -> None:
    decay = 0.14
    assert decay_from_q(500.0, q_from_decay(500.0, decay)) == pytest.approx(decay)
    with pytest.raises(ValueError, match="density_kg_m3"):
        replace(OAK_EFFECTIVE, density_kg_m3=0.0)


def test_radiation_is_size_and_frequency_dependent() -> None:
    tiny_low = radiation_efficiency(100.0, 0.05)
    large_low = radiation_efficiency(100.0, 1.0)
    tiny_high = radiation_efficiency(4_000.0, 0.05)
    assert 0.0 < tiny_low < large_low < 1.0
    assert tiny_high > tiny_low


def test_rng_substreams_are_independent_and_repeatable() -> None:
    contact = event_rng(42, 3, RandomStream.CONTACT).normal(size=16)
    repeated = event_rng(42, 3, RandomStream.CONTACT).normal(size=16)
    roughness = event_rng(42, 3, RandomStream.ROUGHNESS).normal(size=16)
    np.testing.assert_array_equal(contact, repeated)
    assert not np.array_equal(contact, roughness)

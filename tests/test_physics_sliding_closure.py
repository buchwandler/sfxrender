from __future__ import annotations

import numpy as np
import pytest

from sfxrender._physics.closure import render_terminal_closure
from sfxrender._physics.motion import minimum_jerk_motion
from sfxrender._physics.models import FrictionProfile, ModalBody, Mode as SlidingMode
from sfxrender._physics.sliding import render_sliding_source
from sfxrender._printer import generate_printer_model


def _rms(samples: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(samples.astype(np.float64)))))


@pytest.mark.parametrize("sample_rate", [8_000, 24_000, 48_000])
def test_sliding_source_is_deterministic_and_finite(sample_rate: int) -> None:
    motion = minimum_jerk_motion(duration_s=0.55, sample_rate=sample_rate)
    profile = FrictionProfile(
        base_gain=0.55,
        roughness=0.64,
        noise_band_hz=(90.0, 3_200.0),
        stick_strength=0.13,
        slip_strength=0.42,
        f0_hz=(180.0, 480.0),
        harmonic_rolloff=(1.6, 2.2),
        chaos_amount=0.25,
    )
    body = ModalBody(
        modes=(SlidingMode(frequency_hz=420.0, decay_s=0.08, gain=0.3),)
    )
    first = render_sliding_source(
        motion=motion,
        profile=profile,
        sample_rate=sample_rate,
        seed=42,
        event_index=2,
        body=body,
    )
    repeated = render_sliding_source(
        motion=motion,
        profile=profile,
        sample_rate=sample_rate,
        seed=42,
        event_index=2,
        body=body,
    )
    changed = render_sliding_source(
        motion=motion,
        profile=profile,
        sample_rate=sample_rate,
        seed=43,
        event_index=2,
        body=body,
    )

    np.testing.assert_array_equal(first, repeated)
    assert not np.array_equal(first, changed)
    assert first.dtype == np.float32
    assert first.size == motion.position.size
    assert np.isfinite(first).all()


def test_terminal_closure_force_changes_contact_response_without_fake_thump() -> None:
    sample_rate = 24_000
    model = generate_printer_model(seed=42, sample_rate=sample_rate)
    gentle = render_terminal_closure(
        contact=model.tray_stop_contact,
        velocity_m_s=0.22,
        modes=model.tray_modes,
        sample_rate=sample_rate,
        seed=42,
        event_index=3,
    )
    firm = render_terminal_closure(
        contact=model.tray_stop_contact,
        velocity_m_s=0.68,
        modes=model.tray_modes,
        sample_rate=sample_rate,
        seed=42,
        event_index=3,
    )
    no_impact = render_terminal_closure(
        contact=model.tray_stop_contact,
        velocity_m_s=0.0,
        modes=model.tray_modes,
        sample_rate=sample_rate,
        seed=42,
        event_index=3,
    )

    assert _rms(firm) > _rms(gentle) * 1.10
    assert np.count_nonzero(no_impact) == 0

from __future__ import annotations

import numpy as np
import pytest

from sfxrender import SFXRenderer

_DEVICE_URIS = (
    "sfx:electronics.hum?source=transformer&duration=0.8&seed=42",
    "sfx:device.beep?style=soft&pattern=double&seed=42",
    "sfx:phone.notification?style=gentle&count=1&seed=42",
    "sfx:phone.vibrate?duration=0.8&intensity=normal&pattern=steady&surface=wood&seed=42",
    "sfx:device.power_on?device=small&seed=42",
    "sfx:device.power_off?device=small&seed=42",
)


def _rms(samples: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(samples.astype(np.float64)))))


@pytest.mark.parametrize("sample_rate", [8_000, 16_000, 24_000, 48_000])
@pytest.mark.parametrize("uri", _DEVICE_URIS)
def test_device_effects_obey_seeded_pcm_contract(sample_rate: int, uri: str) -> None:
    renderer = SFXRenderer(sample_rate=sample_rate)
    first = renderer.render_uri(uri)
    repeated = renderer.render_uri(uri)
    changed = renderer.render_uri(uri.replace("seed=42", "seed=43"))

    np.testing.assert_array_equal(first.samples, repeated.samples)
    assert not np.array_equal(first.samples, changed.samples)
    assert first.sample_rate == sample_rate
    assert first.samples.dtype == np.float32
    assert first.samples.ndim == 1
    assert first.samples.size > 0
    assert np.isfinite(first.samples).all()
    assert float(np.max(np.abs(first.samples))) <= 1.0
    assert _rms(first.samples) > 1e-5


def test_electronics_hum_has_harmonic_energy_beyond_its_fundamental() -> None:
    sample_rate = 24_000
    hum = (
        SFXRenderer(sample_rate=sample_rate)
        .render_uri("sfx:electronics.hum?source=transformer&duration=2.0&seed=42")
        .samples
    )
    spectrum = np.abs(np.fft.rfft(hum.astype(np.float64)))
    frequencies = np.fft.rfftfreq(hum.size, 1.0 / sample_rate)
    fundamental = spectrum[np.abs(frequencies - 60.0) < 2.0].max()
    harmonics = sum(
        spectrum[np.abs(frequencies - frequency) < 2.0].max()
        for frequency in (120.0, 180.0, 240.0, 300.0)
    )
    assert harmonics > fundamental * 0.3


def test_tone_patterns_and_phone_vibration_are_semantically_distinct() -> None:
    renderer = SFXRenderer(sample_rate=24_000)
    beep = renderer.render_uri("sfx:device.beep?style=soft&pattern=single&seed=42")
    alert = renderer.render_uri("sfx:device.beep?style=alert&pattern=triple&seed=42")
    gentle = renderer.render_uri("sfx:phone.notification?style=gentle&count=1&seed=42")
    urgent = renderer.render_uri("sfx:phone.notification?style=urgent&count=2&seed=42")
    vibration = renderer.render_uri("sfx:phone.vibrate?duration=0.8&surface=wood&seed=42")

    assert alert.duration > beep.duration
    assert urgent.duration > gentle.duration
    assert not np.array_equal(beep.samples, gentle.samples)
    assert not np.array_equal(gentle.samples, vibration.samples)
    assert _rms(vibration.samples) < _rms(gentle.samples)


def test_power_composites_have_distinct_startup_and_shutdown_actions() -> None:
    renderer = SFXRenderer(sample_rate=24_000)
    powered_on = renderer.render_uri("sfx:device.power_on?device=small&seed=42")
    powered_off = renderer.render_uri("sfx:device.power_off?device=small&seed=42")
    appliance = renderer.render_uri("sfx:device.power_on?device=appliance&seed=42")
    assert not np.array_equal(powered_on.samples, powered_off.samples)
    assert appliance.duration > powered_on.duration

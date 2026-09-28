"""Render a fixed-seed gallery of electronic device effects."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from sfxrender import RenderedSound, SFXRenderer, SfxSpec

DEVICE_ITEMS = (
    (
        "electronics-hum-transformer.wav",
        "sfx:electronics.hum?source=transformer&duration=2.0&seed=901",
    ),
    ("device-beep-alert-triple.wav", "sfx:device.beep?style=alert&pattern=triple&seed=902"),
    ("phone-notification-gentle.wav", "sfx:phone.notification?style=gentle&count=1&seed=903"),
    ("phone-notification-urgent.wav", "sfx:phone.notification?style=urgent&count=2&seed=904"),
    (
        "phone-vibrate-wood-pulsed.wav",
        "sfx:phone.vibrate?duration=1.2&intensity=strong&pattern=pulsed&surface=wood&seed=905",
    ),
    ("device-power-on-small.wav", "sfx:device.power_on?device=small&seed=906"),
    ("device-power-off-appliance.wav", "sfx:device.power_off?device=appliance&seed=907"),
)


def main(output_dir: str | Path | None = None) -> None:
    destination = (
        Path(output_dir)
        if output_dir is not None
        else Path(__file__).resolve().parents[1] / "example-artifacts"
    )
    destination.mkdir(parents=True, exist_ok=True)
    renderer = SFXRenderer(sample_rate=24_000)
    sounds: list[RenderedSound] = []
    for filename, uri in DEVICE_ITEMS:
        sound = renderer.render_uri(uri)
        sound.write_wav(destination / filename)
        sounds.append(sound)
        print(f"Rendered {uri} -> {destination / filename}")
    gap = np.zeros(round(0.55 * renderer.sample_rate), dtype=np.float32)
    parts = [
        part
        for index, sound in enumerate(sounds)
        for part in ((gap,) if index else ()) + (sound.samples,)
    ]
    showcase = RenderedSound(
        np.concatenate(parts),
        renderer.sample_rate,
        SfxSpec("example.device_quality_gallery", {}),
    )
    showcase.write_wav(destination / "device-quality-showcase.wav")
    print(f"Wrote combined showcase -> {destination / 'device-quality-showcase.wav'}")


if __name__ == "__main__":
    main()

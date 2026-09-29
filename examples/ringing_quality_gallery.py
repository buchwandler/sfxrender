"""Render fixed-seed phone and doorbell listening comparisons."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from sfxrender import RenderedSound, SFXRenderer, SfxSpec

RINGING_ITEMS = (
    ("phone-classic.wav", "sfx:phone.ring?style=classic&count=2&interval=1.2&seed=21"),
    ("phone-electronic.wav", "sfx:phone.ring?style=electronic&count=2&interval=1.2&seed=21"),
    ("doorbell-chime.wav", "sfx:doorbell.ring?style=chime&count=2&interval=0.8&seed=31"),
    ("doorbell-electronic.wav", "sfx:doorbell.ring?style=electronic&count=2&interval=0.8&seed=31"),
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
    for filename, uri in RINGING_ITEMS:
        sound = renderer.render_uri(uri)
        sound.write_wav(destination / filename)
        sounds.append(sound)
        print(f"Rendered {uri} -> {destination / filename}")

    gap = np.zeros(round(0.5 * renderer.sample_rate), dtype=np.float32)
    parts = [
        part
        for index, sound in enumerate(sounds)
        for part in ((gap,) if index else ()) + (sound.samples,)
    ]
    showcase = RenderedSound(
        np.concatenate(parts),
        renderer.sample_rate,
        SfxSpec("example.ringing_quality_gallery", {}),
    )
    showcase.write_wav(destination / "ringing-quality-showcase.wav")


if __name__ == "__main__":
    main()

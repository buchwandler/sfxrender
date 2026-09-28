"""Render a fixed-seed gallery of wind, weather, fire, and wildlife."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from sfxrender import RenderedSound, SFXRenderer, SfxSpec

ENVIRONMENT_ITEMS = (
    ("wind-leafy-strong.wav", "sfx:wind?intensity=strong&texture=leafy&duration=8.0&seed=1201"),
    (
        "transition-whoosh-rise.wav",
        "sfx:transition.whoosh?style=forceful&direction=rise&duration=1.8&seed=1202",
    ),
    ("rain-heavy-roof.wav", "sfx:rain?intensity=heavy&surface=roof&duration=8.0&seed=1203"),
    ("fire-crackle-active.wav", "sfx:fire.crackle?activity=active&duration=8.0&seed=1204"),
    ("birds-ambience-busy.wav", "sfx:birds.ambience?activity=busy&duration=8.0&seed=1205"),
    ("crickets-ambience-busy.wav", "sfx:crickets.ambience?activity=busy&duration=8.0&seed=1206"),
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
    for filename, uri in ENVIRONMENT_ITEMS:
        sound = renderer.render_uri(uri)
        sound.write_wav(destination / filename)
        sounds.append(sound)
        print(f"Rendered {uri} -> {destination / filename}")
    gap = np.zeros(round(0.75 * renderer.sample_rate), dtype=np.float32)
    parts: list[np.ndarray] = []
    for index, sound in enumerate(sounds):
        if index:
            parts.append(gap)
        parts.append(sound.samples)
    showcase = RenderedSound(
        np.concatenate(parts),
        renderer.sample_rate,
        SfxSpec("example.environment_quality_gallery", {}),
    )
    showcase.write_wav(destination / "environment-quality-showcase.wav")
    print(f"Wrote combined showcase -> {destination / 'environment-quality-showcase.wav'}")


if __name__ == "__main__":
    main()

"""Render a fixed-seed listening gallery for procedural Foley quality."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from sfxrender import RenderedSound, SFXRenderer, SfxSpec

FOLEY_ITEMS = (
    (
        "footsteps-wood-boots.wav",
        "sfx:footsteps.walk?surface=wood&footwear=boots&count=6&force=0.62&interval=0.50&seed=401",
    ),
    (
        "footsteps-wood-shoes.wav",
        "sfx:footsteps.walk?surface=wood&footwear=shoes&count=6&force=0.62&interval=0.50&seed=402",
    ),
    (
        "footsteps-stone-shoes.wav",
        "sfx:footsteps.walk?surface=stone&footwear=shoes&count=6&force=0.62&interval=0.50&seed=403",
    ),
    (
        "footsteps-stone-heels.wav",
        "sfx:footsteps.walk?surface=stone&footwear=heels&count=6&force=0.62&interval=0.50&seed=404",
    ),
    (
        "footsteps-carpet-barefoot.wav",
        "sfx:footsteps.walk?surface=carpet&footwear=barefoot&count=6&force=0.62&interval=0.50&seed=405",
    ),
    (
        "footsteps-gravel-boots.wav",
        "sfx:footsteps.walk?surface=gravel&footwear=boots&count=6&force=0.62&interval=0.50&seed=406",
    ),
    ("knock-wood.wav", "sfx:impact.knock?material=wood&count=3&force=0.68&interval=0.34&seed=411"),
    ("knock-oak.wav", "sfx:impact.knock?material=oak&count=3&force=0.68&interval=0.34&seed=412"),
    ("knock-wall.wav", "sfx:impact.knock?material=wall&count=3&force=0.68&interval=0.34&seed=413"),
    (
        "knock-metal.wav",
        "sfx:impact.knock?material=metal&count=3&force=0.68&interval=0.34&seed=414",
    ),
)


def main(output_dir: str | Path | None = None) -> None:
    """Write the fixed-seed individual Foley WAVs and a spaced comparison strip."""
    destination = (
        Path(output_dir)
        if output_dir is not None
        else Path(__file__).resolve().parents[1] / "example-artifacts"
    )
    destination.mkdir(parents=True, exist_ok=True)
    renderer = SFXRenderer(sample_rate=24_000)
    sounds: list[RenderedSound] = []
    for filename, uri in FOLEY_ITEMS:
        sound = renderer.render_uri(uri)
        sound.write_wav(destination / filename)
        sounds.append(sound)
        print(f"Rendered {uri} -> {destination / filename}")

    gap = np.zeros(int(0.55 * renderer.sample_rate), dtype=np.float32)
    parts: list[np.ndarray] = []
    for index, sound in enumerate(sounds):
        if index:
            parts.append(gap)
        parts.append(sound.samples)
    showcase = RenderedSound(
        np.concatenate(parts),
        renderer.sample_rate,
        SfxSpec("example.foley_quality_showcase", {}),
    )
    showcase_path = destination / "foley-quality-showcase.wav"
    showcase.write_wav(showcase_path)
    print(f"Wrote combined showcase -> {showcase_path}")


if __name__ == "__main__":
    main()

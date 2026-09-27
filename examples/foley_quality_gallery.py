"""Render a fixed-seed listening gallery for procedural Foley quality."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from sfxrender import RenderedSound, SFXRenderer, SfxSpec

FOLEY_ITEMS = (
    (
        "01-footsteps-wood-boots.wav",
        "sfx:footsteps.walk?surface=wood&footwear=boots&count=6&force=0.62&interval=0.50&seed=401",
    ),
    (
        "02-footsteps-wood-shoes.wav",
        "sfx:footsteps.walk?surface=wood&footwear=shoes&count=6&force=0.62&interval=0.50&seed=401",
    ),
    (
        "03-footsteps-stone-shoes.wav",
        "sfx:footsteps.walk?surface=stone&footwear=shoes&count=6&force=0.62&interval=0.50&seed=403",
    ),
    (
        "04-footsteps-stone-heels.wav",
        "sfx:footsteps.walk?surface=stone&footwear=heels&count=6&force=0.62&interval=0.50&seed=403",
    ),
    (
        "05-footsteps-carpet-barefoot.wav",
        "sfx:footsteps.walk?surface=carpet&footwear=barefoot&count=6&force=0.62&interval=0.50&seed=405",
    ),
    (
        "06-footsteps-gravel-boots.wav",
        "sfx:footsteps.walk?surface=gravel&footwear=boots&count=6&force=0.62&interval=0.50&seed=406",
    ),
    (
        "07-footsteps-wood-boots-light-load.wav",
        "sfx:footsteps.walk?surface=wood&footwear=boots&count=6&force=0.35&interval=0.50&seed=407",
    ),
    (
        "08-footsteps-wood-boots-heavy-load.wav",
        "sfx:footsteps.walk?surface=wood&footwear=boots&count=6&force=0.9&interval=0.50&seed=407",
    ),
    ("knock-wood.wav", "sfx:impact.knock?material=wood&count=3&force=0.68&interval=0.34&seed=411"),
    ("knock-oak.wav", "sfx:impact.knock?material=oak&count=3&force=0.68&interval=0.34&seed=411"),
    (
        "knock-oak-fingertip.wav",
        "sfx:impact.knock?material=oak&impactor=fingertip&count=3&force=0.68&interval=0.34&seed=415",
    ),
    (
        "knock-oak-metal-impactor.wav",
        "sfx:impact.knock?material=oak&impactor=metal_object&count=3&force=0.68&interval=0.34&seed=415",
    ),
    ("knock-wall.wav", "sfx:impact.knock?material=wall&count=3&force=0.68&interval=0.34&seed=413"),
    (
        "knock-metal.wav",
        "sfx:impact.knock?material=metal&count=3&force=0.68&interval=0.34&seed=414",
    ),
)


def _write_showcase(
    sounds: list[RenderedSound],
    renderer: SFXRenderer,
    destination: Path,
    filename: str,
    effect_name: str,
) -> None:
    gap = np.zeros(int(0.55 * renderer.sample_rate), dtype=np.float32)
    parts: list[np.ndarray] = []
    for index, sound in enumerate(sounds):
        if index:
            parts.append(gap)
        parts.append(sound.samples)
    showcase = RenderedSound(
        np.concatenate(parts),
        renderer.sample_rate,
        SfxSpec(f"example.{effect_name}", {}),
    )
    output_path = destination / filename
    showcase.write_wav(output_path)
    print(f"Wrote spaced showcase -> {output_path}")


def main(output_dir: str | Path | None = None) -> None:
    """Write fixed-seed individual Foley WAVs and the footstep/Foley strips."""
    destination = (
        Path(output_dir)
        if output_dir is not None
        else Path(__file__).resolve().parents[1] / "example-artifacts"
    )
    destination.mkdir(parents=True, exist_ok=True)
    renderer = SFXRenderer(sample_rate=24_000)
    sounds: list[RenderedSound] = []
    footsteps: list[RenderedSound] = []
    for index, (filename, uri) in enumerate(FOLEY_ITEMS):
        sound = renderer.render_uri(uri)
        sound.write_wav(destination / filename)
        sounds.append(sound)
        if index < 8:
            footsteps.append(sound)
        print(f"Rendered {uri} -> {destination / filename}")
    _write_showcase(
        footsteps,
        renderer,
        destination,
        "foley-footsteps-showcase.wav",
        "foley_footsteps_showcase",
    )
    _write_showcase(
        sounds,
        renderer,
        destination,
        "foley-quality-showcase.wav",
        "foley_quality_showcase",
    )


if __name__ == "__main__":
    main()

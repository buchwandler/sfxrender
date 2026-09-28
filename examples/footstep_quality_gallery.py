"""Render fixed-seed walking, running, and stair-footstep comparisons."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from sfxrender import RenderedSound, SFXRenderer, SfxSpec

FOOTSTEP_ITEMS = (
    (
        "01-walk-wood-shoes.wav",
        "sfx:footsteps.walk?surface=wood&footwear=shoes&count=6&force=0.62&interval=0.50&seed=801",
    ),
    (
        "02-walk-wood-heels.wav",
        "sfx:footsteps.walk?surface=wood&footwear=heels&count=6&force=0.62&interval=0.50&seed=801",
    ),
    (
        "03-walk-wood-barefoot.wav",
        "sfx:footsteps.walk?surface=wood&footwear=barefoot&count=6&force=0.62&interval=0.50&seed=801",
    ),
    (
        "04-walk-stone-boots.wav",
        "sfx:footsteps.walk?surface=stone&footwear=boots&count=6&force=0.62&interval=0.50&seed=804",
    ),
    (
        "05-walk-gravel-shoes.wav",
        "sfx:footsteps.walk?surface=gravel&footwear=shoes&count=6&force=0.62&interval=0.50&seed=805",
    ),
    (
        "06-run-wood-shoes.wav",
        "sfx:footsteps.run?surface=wood&footwear=shoes&count=8&force=0.68&interval=0.32&seed=806",
    ),
    (
        "07-run-gravel-boots.wav",
        "sfx:footsteps.run?surface=gravel&footwear=boots&count=7&force=0.68&interval=0.32&seed=807",
    ),
    (
        "08-stairs-wood-up.wav",
        "sfx:footsteps.stairs?surface=wood&direction=up&footwear=shoes&count=6&seed=808",
    ),
    (
        "09-stairs-stone-down.wav",
        "sfx:footsteps.stairs?surface=stone&direction=down&footwear=boots&count=6&seed=809",
    ),
    (
        "10-walk-wood-slow.wav",
        "sfx:footsteps.walk?surface=wood&footwear=shoes&count=1&force=0.62&interval=0.80&seed=810",
    ),
    (
        "11-walk-wood-fast.wav",
        "sfx:footsteps.walk?surface=wood&footwear=shoes&count=1&force=0.62&interval=0.35&seed=810",
    ),
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
    for filename, uri in FOOTSTEP_ITEMS:
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
        SfxSpec("example.footstep_quality_gallery", {}),
    )
    showcase.write_wav(destination / "footstep-quality-showcase.wav")
    print(f"Wrote combined showcase -> {destination / 'footstep-quality-showcase.wav'}")


if __name__ == "__main__":
    main()

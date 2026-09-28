"""Render fixed-seed walking, running, and stair-footstep comparisons."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from sfxrender import RenderedSound, SFXRenderer, SfxSpec

FOOTSTEP_ITEMS = (
    ("footsteps-run-wood.wav", "sfx:footsteps.run?surface=wood&footwear=shoes&count=8&seed=801"),
    (
        "footsteps-run-gravel.wav",
        "sfx:footsteps.run?surface=gravel&footwear=boots&count=7&seed=802",
    ),
    (
        "footsteps-stairs-wood-up.wav",
        "sfx:footsteps.stairs?surface=wood&direction=up&footwear=shoes&count=6&seed=803",
    ),
    (
        "footsteps-stairs-stone-down.wav",
        "sfx:footsteps.stairs?surface=stone&direction=down&footwear=boots&count=6&seed=804",
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

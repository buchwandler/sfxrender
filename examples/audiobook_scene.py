"""Compose a short fixed-seed Foley scene for audiobook narration."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from sfxrender import RenderedSound, SFXRenderer, SfxSpec


def main(output_dir: str | Path | None = None) -> None:
    """Write a wood-walk, three-knock, and slow door-opening scene."""
    destination = (
        Path(output_dir)
        if output_dir is not None
        else Path(__file__).resolve().parents[1] / "example-artifacts"
    )
    destination.mkdir(parents=True, exist_ok=True)
    renderer = SFXRenderer(sample_rate=24_000)
    events = [
        renderer.render_uri(
            "sfx:footsteps.walk?surface=wood&footwear=boots&count=5&force=0.62&interval=0.48&seed=101"
        ),
        renderer.render_uri(
            "sfx:impact.knock?material=oak&count=3&force=0.68&interval=0.24&seed=102"
        ),
        renderer.render_uri("sfx:door.open?material=wood&speed=slow&creak=0.7&seed=103"),
    ]
    pauses = (0.65, 0.55)
    pieces: list[np.ndarray] = []
    for index, event in enumerate(events):
        if index:
            pieces.append(np.zeros(int(pauses[index - 1] * renderer.sample_rate), dtype=np.float32))
        pieces.append(event.samples)

    scene = RenderedSound(
        np.concatenate(pieces),
        renderer.sample_rate,
        SfxSpec("example.audiobook_scene", {}),
    )
    output = destination / "audiobook-scene-arrival.wav"
    scene.write_wav(output)
    print(f"Wrote audiobook Foley scene -> {output}")


if __name__ == "__main__":
    main()

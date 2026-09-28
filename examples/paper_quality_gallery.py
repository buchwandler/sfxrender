"""Render a fixed-seed paper flutter and handling gallery."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from sfxrender import RenderedSound, SFXRenderer, SfxSpec

PAPER_ITEMS = (
    ("paper-page-turn-slow.wav", "sfx:paper.page_turn?pages=1&speed=slow&seed=701"),
    ("paper-page-turn-fast-stack.wav", "sfx:paper.page_turn?pages=4&speed=fast&seed=702"),
    ("paper-handle-gentle.wav", "sfx:paper.handle?duration=1.2&intensity=gentle&seed=703"),
    ("paper-handle-rough.wav", "sfx:paper.handle?duration=1.2&intensity=rough&seed=704"),
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
    for filename, uri in PAPER_ITEMS:
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
        SfxSpec("example.paper_quality_gallery", {}),
    )
    showcase.write_wav(destination / "paper-quality-showcase.wav")
    print(f"Wrote combined showcase -> {destination / 'paper-quality-showcase.wav'}")


if __name__ == "__main__":
    main()

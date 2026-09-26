"""Render reproducible examples for each built-in SFXRender effect."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from sfxrender import RenderedSound, SFXRenderer, SfxSpec, catalog

CATALOG_ITEMS = (
    (
        "01-knock-oak-3.wav",
        "sfx:impact.knock?material=oak&count=3&force=0.72&interval=0.23&seed=42",
    ),
    (
        "02-knock-metal-2.wav",
        "sfx:impact.knock?material=metal&count=2&force=0.62&interval=0.30&seed=7",
    ),
    (
        "03-footsteps-wood-boots.wav",
        "sfx:footsteps.walk?surface=wood&footwear=boots&count=6&force=0.58&interval=0.52&seed=11",
    ),
    (
        "04-footsteps-gravel-shoes.wav",
        "sfx:footsteps.walk?surface=gravel&footwear=shoes&count=6&force=0.56&interval=0.50&seed=12",
    ),
    (
        "05-phone-classic.wav",
        "sfx:phone.ring?style=classic&count=2&interval=1.20&seed=21",
    ),
    (
        "06-phone-electronic.wav",
        "sfx:phone.ring?style=electronic&count=3&interval=0.72&seed=22",
    ),
    (
        "07-door-wood-slow.wav",
        "sfx:door.open?material=wood&speed=slow&creak=0.72&seed=31",
    ),
    (
        "08-door-metal-fast.wav",
        "sfx:door.open?material=metal&speed=fast&creak=0.55&seed=32",
    ),
)


def main(output_dir: str | Path | None = None) -> None:
    """Render individual effect files and a combined listening showcase."""
    destination = (
        Path(output_dir)
        if output_dir is not None
        else Path(__file__).resolve().parents[1] / "example-artifacts"
    )
    destination.mkdir(parents=True, exist_ok=True)
    renderer = SFXRenderer(sample_rate=24_000)

    for effect, descriptor in catalog().items():
        print(f"{effect}: {descriptor['description']}")

    sounds: list[RenderedSound] = []
    for filename, uri in CATALOG_ITEMS:
        sound = renderer.render_uri(uri)
        sound.write_wav(destination / filename)
        sounds.append(sound)
        print(f"Rendered {uri} -> {destination / filename}")

    gap = np.zeros(int(0.5 * renderer.sample_rate), dtype=np.float32)
    showcase_parts: list[np.ndarray] = []
    for index, sound in enumerate(sounds):
        if index:
            showcase_parts.append(gap)
        showcase_parts.append(sound.samples)
    showcase = RenderedSound(
        np.concatenate(showcase_parts),
        renderer.sample_rate,
        SfxSpec("example.catalog_showcase", {}),
    )
    showcase_path = destination / "00-catalog-showcase.wav"
    showcase.write_wav(showcase_path)
    print(f"Wrote combined showcase -> {showcase_path}")


if __name__ == "__main__":
    main()

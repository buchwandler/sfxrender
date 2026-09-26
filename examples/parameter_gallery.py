"""Render same-seed comparison strips for key semantic parameters."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from sfxrender import RenderedSound, SFXRenderer, SfxSpec

KNOCK_URIS = (
    "sfx:impact.knock?material=wood&count=2&force=0.68&interval=0.42&seed=201",
    "sfx:impact.knock?material=oak&count=2&force=0.68&interval=0.42&seed=201",
    "sfx:impact.knock?material=metal&count=2&force=0.68&interval=0.42&seed=201",
    "sfx:impact.knock?material=wall&count=2&force=0.68&interval=0.42&seed=201",
)
FOOTSTEP_URIS = (
    "sfx:footsteps.walk?surface=wood&footwear=shoes&count=3&force=0.62&interval=0.44&seed=202",
    "sfx:footsteps.walk?surface=stone&footwear=shoes&count=3&force=0.62&interval=0.44&seed=202",
    "sfx:footsteps.walk?surface=gravel&footwear=shoes&count=3&force=0.62&interval=0.44&seed=202",
    "sfx:footsteps.walk?surface=carpet&footwear=shoes&count=3&force=0.62&interval=0.44&seed=202",
)


def _write_strip(
    renderer: SFXRenderer, uris: tuple[str, ...], output: Path, effect_name: str
) -> None:
    gap = np.zeros(int(0.35 * renderer.sample_rate), dtype=np.float32)
    parts: list[np.ndarray] = []
    for index, uri in enumerate(uris):
        if index:
            parts.append(gap)
        parts.append(renderer.render_uri(uri).samples)
    strip = RenderedSound(
        np.concatenate(parts),
        renderer.sample_rate,
        SfxSpec(f"example.{effect_name}", {}),
    )
    strip.write_wav(output)
    print(f"Rendered parameter comparison -> {output}")


def main(output_dir: str | Path | None = None) -> None:
    """Write same-seed A/B strips for knock materials and walking surfaces."""
    destination = (
        Path(output_dir)
        if output_dir is not None
        else Path(__file__).resolve().parents[1] / "example-artifacts"
    )
    destination.mkdir(parents=True, exist_ok=True)
    renderer = SFXRenderer(sample_rate=24_000)
    _write_strip(
        renderer,
        KNOCK_URIS,
        destination / "gallery-knock-materials.wav",
        "knock_material_gallery",
    )
    _write_strip(
        renderer,
        FOOTSTEP_URIS,
        destination / "gallery-footstep-surfaces.wav",
        "footstep_surface_gallery",
    )


if __name__ == "__main__":
    main()

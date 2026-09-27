"""Render a fixed-seed listening gallery for generated door actions."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from sfxrender import RenderedSound, SFXRenderer, SfxSpec

DOOR_ITEMS = (
    (
        "01-door-open-wood-slow-creaky.wav",
        "sfx:door.open?material=wood&speed=slow&creak=0.75&seed=501",
    ),
    (
        "02-door-open-wood-slow-low-creak.wav",
        "sfx:door.open?material=wood&speed=slow&creak=0.1&seed=501",
    ),
    (
        "03-door-open-wood-normal.wav",
        "sfx:door.open?material=wood&speed=normal&creak=0.5&seed=501",
    ),
    (
        "04-door-open-wood-fast.wav",
        "sfx:door.open?material=wood&speed=fast&creak=0.5&seed=501",
    ),
    (
        "05-door-open-metal-normal.wav",
        "sfx:door.open?material=metal&speed=normal&creak=0.55&seed=504",
    ),
    (
        "06-door-close-wood-slow.wav",
        "sfx:door.close?material=wood&speed=slow&force=0.55&creak=0.25&seed=501",
    ),
    (
        "07-door-close-wood-slow-hard.wav",
        "sfx:door.close?material=wood&speed=slow&force=0.9&creak=0.25&seed=501",
    ),
    (
        "08-door-close-wood-normal.wav",
        "sfx:door.close?material=wood&speed=normal&force=0.65&creak=0.25&seed=501",
    ),
    (
        "09-door-close-wood-fast-hard.wav",
        "sfx:door.close?material=wood&speed=fast&force=0.9&creak=0.2&seed=501",
    ),
    (
        "10-door-close-metal-normal.wav",
        "sfx:door.close?material=metal&speed=normal&force=0.7&creak=0.3&seed=507",
    ),
)


def _write_showcase(sounds: list[RenderedSound], renderer: SFXRenderer, path: Path) -> None:
    gap = np.zeros(round(0.55 * renderer.sample_rate), dtype=np.float32)
    parts: list[np.ndarray] = []
    for index, sound in enumerate(sounds):
        if index:
            parts.append(gap)
        parts.append(sound.samples)
    showcase = RenderedSound(
        np.concatenate(parts),
        renderer.sample_rate,
        SfxSpec("example.door_quality_showcase", {}),
    )
    showcase.write_wav(path)


def main(output_dir: str | Path | None = None) -> None:
    """Write individual door WAVs and a spaced combined listening strip."""
    destination = (
        Path(output_dir)
        if output_dir is not None
        else Path(__file__).resolve().parents[1] / "example-artifacts"
    )
    destination.mkdir(parents=True, exist_ok=True)
    renderer = SFXRenderer(sample_rate=24_000)
    sounds: list[RenderedSound] = []
    for filename, uri in DOOR_ITEMS:
        sound = renderer.render_uri(uri)
        sound.write_wav(destination / filename)
        sounds.append(sound)
        print(f"Rendered {uri} -> {destination / filename}")
    showcase_path = destination / "door-quality-showcase.wav"
    _write_showcase(sounds, renderer, showcase_path)
    print(f"Wrote combined showcase -> {showcase_path}")


if __name__ == "__main__":
    main()

"""Render a fixed-seed gallery of indoor and urban ambience."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from sfxrender import RenderedSound, SFXRenderer, SfxSpec

AMBIENCE_ITEMS = (
    ("room-tone-quiet.wav", "sfx:room_tone?character=quiet&duration=6.0&seed=1101"),
    ("room-tone-electrical.wav", "sfx:room_tone?character=electrical&duration=6.0&seed=1102"),
    ("office-ambience-quiet.wav", "sfx:office.ambience?activity=quiet&duration=10.0&seed=1103"),
    ("office-ambience-busy.wav", "sfx:office.ambience?activity=busy&duration=10.0&seed=1104"),
    ("city-ambience-calm.wav", "sfx:city.ambience?activity=calm&duration=16.0&seed=1105"),
    ("city-ambience-busy.wav", "sfx:city.ambience?activity=busy&duration=16.0&seed=1106"),
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
    for filename, uri in AMBIENCE_ITEMS:
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
        SfxSpec("example.ambience_quality_gallery", {}),
    )
    showcase.write_wav(destination / "ambience-quality-showcase.wav")
    print(f"Wrote combined showcase -> {destination / 'ambience-quality-showcase.wav'}")


if __name__ == "__main__":
    main()

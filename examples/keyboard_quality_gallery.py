"""Render fixed-seed keyboard typing pace and force comparisons."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from sfxrender import RenderedSound, SFXRenderer, SfxSpec

KEYBOARD_ITEMS = (
    (
        "keyboard-slow-light.wav",
        "sfx:keyboard.typing?speed=slow&force=light&duration=5.0&seed=1306",
    ),
    (
        "keyboard-steady-light.wav",
        "sfx:keyboard.typing?speed=steady&force=light&duration=5.0&seed=1306",
    ),
    (
        "keyboard-fast-light.wav",
        "sfx:keyboard.typing?speed=fast&force=light&duration=5.0&seed=1306",
    ),
    (
        "keyboard-fast-firm.wav",
        "sfx:keyboard.typing?speed=fast&force=firm&duration=5.0&seed=1306",
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
    for filename, uri in KEYBOARD_ITEMS:
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
        SfxSpec("example.keyboard_quality_gallery", {}),
    )
    showcase.write_wav(destination / "keyboard-quality-showcase.wav")
    print(f"Wrote combined showcase -> {destination / 'keyboard-quality-showcase.wav'}")


if __name__ == "__main__":
    main()

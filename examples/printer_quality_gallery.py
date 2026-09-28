"""Render a fixed-seed listening gallery for printer and pen story sounds."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from sfxrender import RenderedSound, SFXRenderer, SfxSpec

PRINTER_ITEMS = (
    (
        "01-printer-print-one-normal.wav",
        "sfx:printer.print?pages=1&speed=normal&seed=501",
    ),
    (
        "02-printer-print-three-normal.wav",
        "sfx:printer.print?pages=3&speed=normal&seed=502",
    ),
    (
        "03-printer-print-one-fast.wav",
        "sfx:printer.print?pages=1&speed=fast&seed=503",
    ),
    (
        "04-printer-tray-open-full.wav",
        "sfx:printer.tray_open?speed=normal&paper_load=full&seed=504",
    ),
    (
        "05-printer-tray-open-empty.wav",
        "sfx:printer.tray_open?speed=normal&paper_load=empty&seed=505",
    ),
    (
        "06-printer-tray-close-gentle-empty.wav",
        "sfx:printer.tray_close?speed=normal&paper_load=empty&force=gentle&seed=506",
    ),
    (
        "07-printer-tray-close-firm-full.wav",
        "sfx:printer.tray_close?speed=normal&paper_load=full&force=firm&seed=507",
    ),
    ("08-printer-power-off.wav", "sfx:printer.power_switch?state=off&seed=508"),
    ("09-printer-power-on.wav", "sfx:printer.power_switch?state=on&seed=509"),
    ("10-printer-restart-normal.wav", "sfx:printer.restart?speed=normal&seed=510"),
    ("11-printer-wake-light.wav", "sfx:printer.wake?depth=light&seed=511"),
    (
        "12-pen-write-normal.wav",
        "sfx:pen.write?duration=1.6&speed=normal&pressure=0.55&seed=510",
    ),
    (
        "13-pen-write-fast-light.wav",
        "sfx:pen.write?duration=1.4&speed=fast&pressure=0.25&seed=513",
    ),
    (
        "14-pen-write-slow-heavy.wav",
        "sfx:pen.write?duration=1.4&speed=slow&pressure=0.9&seed=512",
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
        SfxSpec("example.printer_story_sfx_showcase", {}),
    )
    showcase.write_wav(path)


def main(output_dir: str | Path | None = None) -> None:
    """Write individual printer/pen WAVs and one combined listening strip."""
    destination = (
        Path(output_dir)
        if output_dir is not None
        else Path(__file__).resolve().parents[1] / "example-artifacts"
    )
    destination.mkdir(parents=True, exist_ok=True)
    renderer = SFXRenderer(sample_rate=24_000)
    sounds: list[RenderedSound] = []
    for filename, uri in PRINTER_ITEMS:
        sound = renderer.render_uri(uri)
        sound.write_wav(destination / filename)
        sounds.append(sound)
        print(f"Rendered {uri} -> {destination / filename}")
    showcase_path = destination / "printer-story-sfx-showcase.wav"
    _write_showcase(sounds, renderer, showcase_path)
    print(f"Wrote combined showcase -> {showcase_path}")


if __name__ == "__main__":
    main()

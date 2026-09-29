"""Render focused keyboard contacts and fixed-seed typing pace/force comparisons."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from sfxrender import RenderedSound, SFXRenderer, SfxSpec
from sfxrender._keyboard import KeyEvent, _render_keyboard_events

SAMPLE_RATE = 24_000
SEED = 1306
_GAP_SECONDS = 0.55
_SEQUENCE_ITEMS = (
    ("08-slow-light.wav", "sfx:keyboard.typing?speed=slow&force=light&duration=5&seed=1306"),
    ("09-steady-light.wav", "sfx:keyboard.typing?speed=steady&force=light&duration=5&seed=1306"),
    ("10-fast-light.wav", "sfx:keyboard.typing?speed=fast&force=light&duration=5&seed=1306"),
    ("11-fast-firm.wav", "sfx:keyboard.typing?speed=fast&force=firm&duration=5&seed=1306"),
)


def _isolated_sound(
    key_class: str = "normal",
    *,
    variant: int = 0,
    velocity: float = 0.115,
    release_time_s: float = 0.16,
) -> RenderedSound:
    event = KeyEvent(0.05, release_time_s, key_class, variant, velocity, 0.5)
    samples = _render_keyboard_events(
        (event,),
        duration=0.22,
        sample_rate=SAMPLE_RATE,
        seed=SEED,
    )
    return RenderedSound(samples, SAMPLE_RATE, SfxSpec("example.keyboard_quality_gallery", {}))


def _variant_strip() -> RenderedSound:
    gap = np.zeros(round(0.12 * SAMPLE_RATE), dtype=np.float32)
    parts = [
        part
        for index in range(5)
        for part in ((gap,) if index else ())
        + (_isolated_sound(variant=index).samples,)
    ]
    return RenderedSound(
        np.concatenate(parts), SAMPLE_RATE, SfxSpec("example.keyboard_quality_gallery", {})
    )


def _release_focused() -> RenderedSound:
    full = _isolated_sound(release_time_s=0.16)
    release_start = round(0.16 * SAMPLE_RATE)
    return RenderedSound(
        full.samples[release_start:].copy(),
        SAMPLE_RATE,
        SfxSpec("example.keyboard_quality_gallery", {}),
    )


def _gallery_items(renderer: SFXRenderer) -> tuple[tuple[str, RenderedSound], ...]:
    items = [
        ("01-normal-light-single.wav", _isolated_sound()),
        ("02-normal-firm-single.wav", _isolated_sound(velocity=0.205)),
        ("03-normal-variant-strip.wav", _variant_strip()),
        ("04-space-single.wav", _isolated_sound("space")),
        ("05-enter-single.wav", _isolated_sound("enter")),
        ("06-backspace-single.wav", _isolated_sound("backspace")),
        ("07-release-focused.wav", _release_focused()),
    ]
    items.extend(
        (filename, renderer.render_uri(uri)) for filename, uri in _SEQUENCE_ITEMS
    )
    return tuple(items)


def main(output_dir: str | Path | None = None) -> None:
    destination = (
        Path(output_dir)
        if output_dir is not None
        else Path(__file__).resolve().parents[1] / "example-artifacts"
    )
    destination.mkdir(parents=True, exist_ok=True)
    renderer = SFXRenderer(sample_rate=SAMPLE_RATE)
    items = _gallery_items(renderer)
    gap = np.zeros(round(_GAP_SECONDS * SAMPLE_RATE), dtype=np.float32)
    parts: list[np.ndarray] = []
    for index, (filename, sound) in enumerate(items):
        output_path = destination / filename
        sound.write_wav(output_path)
        print(f"Rendered {filename} -> {output_path}")
        if index:
            parts.append(gap)
        parts.append(sound.samples)
    showcase = RenderedSound(
        np.concatenate(parts),
        SAMPLE_RATE,
        SfxSpec("example.keyboard_quality_gallery", {}),
    )
    showcase_path = destination / "keyboard-quality-showcase.wav"
    showcase.write_wav(showcase_path)
    print(f"Wrote combined showcase -> {showcase_path}")


if __name__ == "__main__":
    main()

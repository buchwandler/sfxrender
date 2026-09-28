"""Render a fixed-seed gallery of the remaining Top-50 effects."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from sfxrender import RenderedSound, SFXRenderer, SfxSpec

TOP50_ITEMS = (
    ("keys-jingle.wav", "sfx:keys.jingle?style=full&duration=1.5&seed=1301"),
    ("lock-turn-deadbolt.wav", "sfx:lock.turn?style=deadbolt&force=firm&duration=1.6&seed=1302"),
    (
        "chair-move-stone.wav",
        "sfx:chair.move?surface=stone&effort=firm&action=set_down&duration=2.3&seed=1303",
    ),
    (
        "cloth-rustle-silk.wav",
        "sfx:cloth.rustle?fabric=silk&activity=active&duration=3.0&seed=1304",
    ),
    ("floor-creak-heavy.wav", "sfx:floor.creak?surface=wood&weight=heavy&duration=1.4&seed=1305"),
    (
        "keyboard-typing-fast.wav",
        "sfx:keyboard.typing?speed=fast&force=light&duration=5.0&seed=1306",
    ),
    ("clock-tick-mantel.wav", "sfx:clock.tick?style=mantel&rate=normal&duration=4.0&seed=1307"),
    (
        "alarm-ring-electronic.wav",
        "sfx:alarm.ring?style=electronic&pattern=intermittent&duration=4.0&seed=1308",
    ),
    (
        "elevator-arrive-large.wav",
        "sfx:elevator.arrive?size=large&chime=double&duration=4.0&seed=1309",
    ),
    ("water-pour-metal.wav", "sfx:water.pour?flow=strong&vessel=metal&duration=3.0&seed=1310"),
    ("water-running-strong.wav", "sfx:water.running?flow=strong&duration=4.0&seed=1311"),
    ("crowd-murmur-busy.wav", "sfx:crowd.murmur?density=busy&duration=5.0&seed=1312"),
    ("thunder-near.wav", "sfx:thunder?intensity=strong&distance=near&duration=6.0&seed=1313"),
    ("car-door-close.wav", "sfx:car.door?action=close&size=suv&force=firm&duration=1.5&seed=1314"),
    ("car-engine-rev.wav", "sfx:car.engine?action=rev&vehicle=truck&duration=4.0&seed=1315"),
    ("car-passby-fast.wav", "sfx:car.passby?speed=fast&vehicle=truck&duration=4.0&seed=1316"),
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
    for filename, uri in TOP50_ITEMS:
        sound = renderer.render_uri(uri)
        sound.write_wav(destination / filename)
        sounds.append(sound)
        print(f"Rendered {uri} -> {destination / filename}")
    gap = np.zeros(round(0.65 * renderer.sample_rate), dtype=np.float32)
    parts: list[np.ndarray] = []
    for index, sound in enumerate(sounds):
        if index:
            parts.append(gap)
        parts.append(sound.samples)
    showcase = RenderedSound(
        np.concatenate(parts),
        renderer.sample_rate,
        SfxSpec("example.top50_quality_gallery", {}),
    )
    showcase.write_wav(destination / "top50-quality-showcase.wav")
    print(f"Wrote combined showcase -> {destination / 'top50-quality-showcase.wav'}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import runpy
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run_example(script: str, output_dir: Path) -> None:
    namespace = runpy.run_path(str(ROOT / "examples" / script))
    namespace["main"](output_dir)


def test_examples_write_reproducible_wavs_only_to_requested_directory(tmp_path: Path) -> None:
    scripts = {
        "render_catalog.py": [
            "00-catalog-showcase.wav",
            "01-knock-oak-3.wav",
            "02-knock-metal-2.wav",
            "03-footsteps-wood-boots.wav",
            "04-footsteps-gravel-shoes.wav",
            "05-phone-classic.wav",
            "06-phone-electronic.wav",
            "07-door-wood-slow.wav",
            "08-door-metal-fast.wav",
        ],
        "audiobook_scene.py": ["audiobook-scene-arrival.wav"],
        "parameter_gallery.py": [
            "gallery-knock-materials.wav",
            "gallery-footstep-surfaces.wav",
        ],
    }
    root_wavs_before = set(ROOT.glob("*.wav"))

    for script, filenames in scripts.items():
        first_dir = tmp_path / script.removesuffix(".py")
        second_dir = tmp_path / f"repeat-{script.removesuffix('.py')}"
        _run_example(script, first_dir)
        _run_example(script, second_dir)
        for filename in filenames:
            first = first_dir / filename
            second = second_dir / filename
            assert first.is_file() and first.stat().st_size > 44
            assert first.read_bytes() == second.read_bytes()

    showcase = tmp_path / "render_catalog" / "00-catalog-showcase.wav"
    with wave.open(str(showcase), "rb") as audio:
        assert audio.getnframes() > 0
        assert audio.getnchannels() == 1
    assert set(ROOT.glob("*.wav")) == root_wavs_before

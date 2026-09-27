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
            "09-door-close-wood-normal.wav",
        ],
        "audiobook_scene.py": ["audiobook-scene-arrival.wav"],
        "parameter_gallery.py": [
            "gallery-knock-materials.wav",
            "gallery-footstep-surfaces.wav",
        ],
        "foley_quality_gallery.py": [
            "01-footsteps-wood-boots.wav",
            "02-footsteps-wood-shoes.wav",
            "03-footsteps-stone-shoes.wav",
            "04-footsteps-stone-heels.wav",
            "05-footsteps-carpet-barefoot.wav",
            "06-footsteps-gravel-boots.wav",
            "knock-wood.wav",
            "knock-oak.wav",
            "knock-wall.wav",
            "knock-metal.wav",
            "foley-footsteps-showcase.wav",
            "foley-quality-showcase.wav",
        ],
        "door_quality_gallery.py": [
            "01-door-open-wood-slow-creaky.wav",
            "02-door-open-wood-normal.wav",
            "03-door-open-wood-fast.wav",
            "04-door-open-metal-normal.wav",
            "05-door-close-wood-slow.wav",
            "06-door-close-wood-normal.wav",
            "07-door-close-wood-fast-hard.wav",
            "08-door-close-metal-normal.wav",
            "door-quality-showcase.wav",
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

    gallery_names = scripts["foley_quality_gallery.py"]
    gallery_dir = tmp_path / "foley_quality_gallery"
    individual_frames = 0
    for filename in gallery_names[:-2]:
        with wave.open(str(gallery_dir / filename), "rb") as audio:
            individual_frames += audio.getnframes()
    with wave.open(str(gallery_dir / gallery_names[-1]), "rb") as audio:
        assert audio.getnchannels() == 1
        assert audio.getframerate() == 24_000
        gap_frames = int(0.55 * audio.getframerate())
        assert audio.getnframes() == individual_frames + gap_frames * (len(gallery_names) - 3)

    footstep_frames = 0
    for filename in gallery_names[:6]:
        with wave.open(str(gallery_dir / filename), "rb") as audio:
            footstep_frames += audio.getnframes()
    with wave.open(str(gallery_dir / gallery_names[-2]), "rb") as audio:
        assert audio.getnframes() == footstep_frames + gap_frames * 5


def test_analyze_foley_writes_deterministic_metrics_to_requested_directory(
    tmp_path: Path,
) -> None:
    root_wavs_before = set(ROOT.glob("*.wav"))
    first_dir = tmp_path / "analysis"
    second_dir = tmp_path / "repeat-analysis"
    _run_example("analyze_foley.py", first_dir)
    _run_example("analyze_foley.py", second_dir)
    report = first_dir / "foley-analysis.txt"
    repeated = second_dir / "foley-analysis.txt"
    assert report.is_file()
    assert report.read_bytes() == repeated.read_bytes()
    content = report.read_text(encoding="utf-8")
    for label in (
        "wood-boots",
        "wood-shoes",
        "stone-shoes",
        "stone-heels",
        "carpet-barefoot",
        "gravel-boots",
    ):
        assert label in content
    for metric in (
        "duration_s",
        "peak",
        "rms",
        "centroid_hz",
        "fraction_80_200_hz",
        "early_rms",
        "mid_rms",
        "tail_rms",
    ):
        assert metric in content.splitlines()[0]
    assert set(ROOT.glob("*.wav")) == root_wavs_before


def test_analyze_doors_writes_waveform_and_physical_metrics(tmp_path: Path) -> None:
    first_dir = tmp_path / "doors-analysis"
    second_dir = tmp_path / "repeat-doors-analysis"
    _run_example("analyze_doors.py", first_dir)
    _run_example("analyze_doors.py", second_dir)
    report = first_dir / "door-analysis.txt"
    repeated = second_dir / "door-analysis.txt"
    assert report.is_file()
    assert report.read_bytes() == repeated.read_bytes()
    content = report.read_text(encoding="utf-8")
    for label in ("open-wood-slow", "close-wood-slow", "close-metal-normal"):
        assert label in content
    for metric in (
        "spectral_centroid_hz",
        "spectral_flatness",
        "fraction_40_250_hz",
        "fraction_5000_8000_hz",
        "early_rms",
        "motion_rms",
        "terminal_rms",
        "frame_rms_p25",
        "frame_rms_p90",
        "intermittency_ratio",
        "panel_mode_count",
        "frame_mode_count",
        "hinge_region_count",
        "hinge_f0_min_hz",
        "hinge_f0_max_hz",
    ):
        assert metric in content.splitlines()[0]

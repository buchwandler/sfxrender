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
            "10-printer-print-1.wav",
            "11-printer-tray-open.wav",
            "12-printer-power-off.wav",
            "13-printer-restart.wav",
            "14-pen-write.wav",
            "15-printer-wake.wav",
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
            "07-footsteps-wood-boots-light-load.wav",
            "08-footsteps-wood-boots-heavy-load.wav",
            "knock-wood.wav",
            "knock-oak.wav",
            "knock-oak-fingertip.wav",
            "knock-oak-metal-impactor.wav",
            "knock-wall.wav",
            "knock-metal.wav",
            "foley-footsteps-showcase.wav",
            "foley-quality-showcase.wav",
        ],
        "footstep_quality_gallery.py": [
            "01-walk-wood-shoes.wav",
            "02-walk-wood-heels.wav",
            "03-walk-wood-barefoot.wav",
            "04-walk-stone-boots.wav",
            "05-walk-gravel-shoes.wav",
            "06-run-wood-shoes.wav",
            "07-run-gravel-boots.wav",
            "08-stairs-wood-up.wav",
            "09-stairs-stone-down.wav",
            "10-walk-wood-slow.wav",
            "11-walk-wood-fast.wav",
            "footstep-quality-showcase.wav",
        ],
        "keyboard_quality_gallery.py": [
            "keyboard-slow-light.wav",
            "keyboard-steady-light.wav",
            "keyboard-fast-light.wav",
            "keyboard-fast-firm.wav",
            "keyboard-quality-showcase.wav",
        ],
        "door_quality_gallery.py": [
            "01-door-open-wood-slow-creaky.wav",
            "02-door-open-wood-slow-low-creak.wav",
            "03-door-open-wood-normal.wav",
            "04-door-open-wood-fast.wav",
            "05-door-open-metal-normal.wav",
            "06-door-close-wood-slow.wav",
            "07-door-close-wood-slow-hard.wav",
            "08-door-close-wood-normal.wav",
            "09-door-close-wood-fast-hard.wav",
            "10-door-close-metal-normal.wav",
            "door-quality-showcase.wav",
        ],
        "printer_quality_gallery.py": [
            "01-printer-print-one-normal.wav",
            "02-printer-print-three-normal.wav",
            "03-printer-print-one-fast.wav",
            "04-printer-tray-open-full.wav",
            "05-printer-tray-open-empty.wav",
            "06-printer-tray-close-gentle-empty.wav",
            "07-printer-tray-close-firm-full.wav",
            "08-printer-power-off.wav",
            "09-printer-power-on.wav",
            "10-printer-restart-normal.wav",
            "11-printer-wake-light.wav",
            "12-pen-write-normal.wav",
            "13-pen-write-fast-light.wav",
            "14-pen-write-slow-heavy.wav",
            "printer-story-sfx-showcase.wav",
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
    for filename in gallery_names[:8]:
        with wave.open(str(gallery_dir / filename), "rb") as audio:
            footstep_frames += audio.getnframes()
    with wave.open(str(gallery_dir / gallery_names[-2]), "rb") as audio:
        assert audio.getnframes() == footstep_frames + gap_frames * 7

    pace_names = scripts["footstep_quality_gallery.py"]
    pace_dir = tmp_path / "footstep_quality_gallery"
    pace_frames = 0
    for filename in pace_names[:-1]:
        with wave.open(str(pace_dir / filename), "rb") as audio:
            pace_frames += audio.getnframes()
    with wave.open(str(pace_dir / pace_names[-1]), "rb") as audio:
        assert audio.getnframes() == pace_frames + gap_frames * (len(pace_names) - 2)
    assert (pace_dir / "10-walk-wood-slow.wav").read_bytes() != (
        pace_dir / "11-walk-wood-fast.wav"
    ).read_bytes()

    keyboard_names = scripts["keyboard_quality_gallery.py"]
    keyboard_dir = tmp_path / "keyboard_quality_gallery"
    keyboard_frames = 0
    for filename in keyboard_names[:-1]:
        with wave.open(str(keyboard_dir / filename), "rb") as audio:
            keyboard_frames += audio.getnframes()
    with wave.open(str(keyboard_dir / keyboard_names[-1]), "rb") as audio:
        assert audio.getnframes() == keyboard_frames + gap_frames * (len(keyboard_names) - 2)


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
        "wood-boots-light-load",
        "wood-boots-heavy-load",
        "carpet-barefoot",
        "gravel-boots",
    ):
        assert label in content
    for metric in (
        "duration_s",
        "onset_time_s",
        "attack_duration_s",
        "active_duration_s",
        "event_density_hz",
        "spectral_flux",
        "dominant_peak_hz",
        "dominant_peak_decay_s",
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
    for label in (
        "open-wood-slow",
        "open-wood-slow-low-creak",
        "close-wood-slow",
        "close-wood-slow-hard",
        "close-metal-normal",
    ):
        assert label in content
    for metric in (
        "spectral_centroid_hz",
        "onset_time_s",
        "attack_duration_s",
        "active_duration_s",
        "event_density_hz",
        "spectral_flux",
        "centroid_early_hz",
        "centroid_middle_hz",
        "centroid_late_hz",
        "dominant_peak_hz",
        "dominant_peak_decay_s",
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


def test_printer_quality_gallery_combined_wav_contains_every_item(tmp_path: Path) -> None:
    root_wavs_before = set(ROOT.glob("*.wav"))
    output_dir = tmp_path / "printer-gallery"
    _run_example("printer_quality_gallery.py", output_dir)
    items = (
        "01-printer-print-one-normal.wav",
        "02-printer-print-three-normal.wav",
        "03-printer-print-one-fast.wav",
        "04-printer-tray-open-full.wav",
        "05-printer-tray-open-empty.wav",
        "06-printer-tray-close-gentle-empty.wav",
        "07-printer-tray-close-firm-full.wav",
        "08-printer-power-off.wav",
        "09-printer-power-on.wav",
        "10-printer-restart-normal.wav",
        "11-printer-wake-light.wav",
        "12-pen-write-normal.wav",
        "13-pen-write-fast-light.wav",
        "14-pen-write-slow-heavy.wav",
    )
    item_frames = 0
    for filename in items:
        with wave.open(str(output_dir / filename), "rb") as audio:
            item_frames += audio.getnframes()
    with wave.open(str(output_dir / "printer-story-sfx-showcase.wav"), "rb") as audio:
        assert audio.getnchannels() == 1
        assert audio.getframerate() == 24_000
        gap_frames = round(0.55 * audio.getframerate())
        assert audio.getnframes() == item_frames + gap_frames * (len(items) - 1)
    assert set(ROOT.glob("*.wav")) == root_wavs_before


def test_printer_story_uses_all_new_semantic_sfx_families() -> None:
    story = (ROOT / "examples" / "printer_story.ssmd").read_text(encoding="utf-8")
    for effect in (
        "printer.print",
        "printer.tray_open",
        "printer.power_switch",
        "printer.restart",
        "printer.wake",
        "pen.write",
    ):
        assert f"sfx:{effect}?" in story
    for seed in ("301", "302", "303", "304", "305", "306", "311"):
        assert f"seed={seed}" in story
    assert story.count('src="sfx:') == 8

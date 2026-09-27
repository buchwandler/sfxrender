"""Smoke tests for the NumPy-only reference fitting and feature comparison CLIs."""

from __future__ import annotations

import json
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

from tools._reference_audio import analyze_signal, load_wav
from tools.fit_modal_reference import fit_references

ROOT = Path(__file__).resolve().parents[1]


def _reference_hit(sample_rate: int, *, frequency_hz: float, scale: float = 1.0) -> np.ndarray:
    size = round(0.65 * sample_rate)
    onset = round(0.03 * sample_rate)
    time = np.arange(size - onset, dtype=np.float64) / sample_rate
    decay = scale * (
        0.62 * np.exp(-time / 0.20) * np.sin(2.0 * np.pi * frequency_hz * time)
        + 0.18 * np.exp(-time / 0.12) * np.sin(2.0 * np.pi * 960.0 * time + 0.31)
    )
    samples = np.zeros(size, dtype=np.float64)
    samples[onset:] = decay
    return samples


def _write_pcm16(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    pcm = np.asarray(np.clip(samples, -1.0, 1.0) * 32_767.0, dtype="<i2")
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(sample_rate)
        stream.writeframes(pcm.tobytes())


def test_reference_analysis_and_modal_fit_are_reproducible(tmp_path: Path) -> None:
    sample_rate = 16_000
    first = tmp_path / "impact-a.wav"
    second = tmp_path / "impact-b.wav"
    _write_pcm16(first, _reference_hit(sample_rate, frequency_hz=440.0), sample_rate)
    _write_pcm16(second, _reference_hit(sample_rate, frequency_hz=441.0, scale=0.8), sample_rate)
    samples, decoded_rate = load_wav(first)
    assert decoded_rate == sample_rate
    assert samples.dtype == np.float64
    features = analyze_signal(samples, decoded_rate)
    assert abs(features["onset_time_s"] - 0.03) < 0.01
    assert features["active_duration_s"] > 0.4
    assert features["spectral_centroid_hz"] > 300.0
    assert all(value > 0.0 for value in features["spectral_centroid_trajectory_hz"])
    assert features["band_energy_fraction"]["250_1000_hz"] > 0.1
    assert features["dominant_peaks"]
    fitted = fit_references(
        [first, second],
        name="wood_sample",
        maximum_modes=8,
        minimum_frequency_hz=100.0,
        maximum_frequency_hz=2_000.0,
        decay_start_ms=100.0,
        decay_duration_ms=400.0,
    )
    repeated = fit_references(
        [first, second],
        name="wood_sample",
        maximum_modes=8,
        minimum_frequency_hz=100.0,
        maximum_frequency_hz=2_000.0,
        decay_start_ms=100.0,
        decay_duration_ms=400.0,
    )
    assert fitted == repeated
    assert fitted["schema"] == "sfxrender.modal_reference.v1"
    assert fitted["recording_count"] == 2
    assert any(abs(mode["frequency_hz"] - 440.5) < 5.0 for mode in fitted["modes"])
    mode = min(fitted["modes"], key=lambda item: abs(item["frequency_hz"] - 440.5))
    assert mode["decay_s"] == pytest.approx(0.20, rel=0.30)
    assert mode["support"] == 2


def test_reference_tool_command_line_writes_json_reports(tmp_path: Path) -> None:
    sample_rate = 16_000
    reference = tmp_path / "reference.wav"
    candidate = tmp_path / "candidate.wav"
    preset_path = tmp_path / "preset.json"
    report_path = tmp_path / "comparison.json"
    _write_pcm16(reference, _reference_hit(sample_rate, frequency_hz=440.0), sample_rate)
    _write_pcm16(candidate, _reference_hit(sample_rate, frequency_hz=442.0, scale=0.7), sample_rate)
    fit_script = ROOT / "tools" / "fit_modal_reference.py"
    compare_script = ROOT / "tools" / "compare_reference.py"
    for script in (fit_script, compare_script):
        help_result = subprocess.run(
            [sys.executable, str(script), "--help"],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        assert help_result.returncode == 0
        assert "usage:" in help_result.stdout.lower()
    fit_result = subprocess.run(
        [
            sys.executable,
            str(fit_script),
            "--name",
            "sample",
            "--output",
            str(preset_path),
            str(reference),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert fit_result.returncode == 0, fit_result.stderr
    compare_result = subprocess.run(
        [
            sys.executable,
            str(compare_script),
            "--reference",
            str(reference),
            "--candidate",
            str(candidate),
            "--output",
            str(report_path),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert compare_result.returncode == 0, compare_result.stderr
    preset = json.loads(preset_path.read_text(encoding="utf-8"))
    comparison = json.loads(report_path.read_text(encoding="utf-8"))
    assert preset["name"] == "sample"
    assert comparison["schema"] == "sfxrender.reference_comparison.v1"
    assert (
        comparison["feature_deltas"]["rms"]["candidate_median"]
        < comparison["feature_deltas"]["rms"]["reference_median"]
    )
    assert comparison["matched_modal_peaks"]
    assert comparison["reference"]["feature_spread"]["rms"]["std"] == 0.0
    assert len(comparison["spectral_centroid_trajectory"]["delta_hz"]) == 5

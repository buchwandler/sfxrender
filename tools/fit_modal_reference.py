"""Fit repeatable modal frequencies and decay estimates from PCM WAV references."""

from __future__ import annotations

import argparse
import json
import math
import sys
import wave
from pathlib import Path
from statistics import median
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools._reference_audio import analyze_signal, estimate_decay_s, load_wav


def _cluster_peaks(
    recordings: list[dict[str, Any]],
    *,
    maximum_modes: int,
    minimum_support_fraction: float = 0.4,
) -> list[dict[str, float | int]]:
    clusters: list[list[tuple[int, float, float, float]]] = []
    for recording_index, recording in enumerate(recordings):
        for peak in recording["analysis"]["dominant_peaks"]:
            frequency = float(peak["frequency_hz"])
            decay_estimate = peak["decay_s"]
            if decay_estimate is None:
                continue
            decay = float(decay_estimate)
            if decay <= 0.0:
                continue
            best: list[tuple[int, float, float, float]] | None = None
            best_distance = math.inf
            for cluster in clusters:
                center = median(item[1] for item in cluster)
                tolerance = max(8.0, center * 0.018)
                distance = abs(frequency - center)
                if (
                    distance <= tolerance
                    and distance < best_distance
                    and all(item[0] != recording_index for item in cluster)
                ):
                    best, best_distance = cluster, distance
            observation = (
                recording_index,
                frequency,
                decay,
                float(peak["relative_amplitude"]),
            )
            if best is None:
                clusters.append([observation])
            else:
                best.append(observation)
    minimum_support = max(1, math.ceil(len(recordings) * minimum_support_fraction))
    fitted: list[dict[str, float | int]] = []
    for cluster in clusters:
        if len(cluster) < minimum_support:
            continue
        frequencies = [item[1] for item in cluster]
        decays = [item[2] for item in cluster]
        gains = [item[3] for item in cluster]
        frequency = float(np_median(frequencies))
        decay = float(np_median(decays))
        fitted.append(
            {
                "frequency_hz": frequency,
                "decay_s": decay,
                "q": math.pi * frequency * decay,
                "input_gain": float(np_median(gains)),
                "support": len(cluster),
                "frequency_std_hz": float(_std(frequencies)),
                "decay_std_s": float(_std(decays)),
            }
        )
    fitted.sort(key=lambda mode: float(mode["input_gain"]), reverse=True)
    selected = fitted[:maximum_modes]
    return sorted(selected, key=lambda mode: float(mode["frequency_hz"]))


def np_median(values: list[float]) -> float:
    return float(median(values))


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))


def fit_references(
    paths: list[Path],
    *,
    name: str,
    maximum_modes: int,
    minimum_frequency_hz: float,
    maximum_frequency_hz: float,
    decay_start_ms: float,
    decay_duration_ms: float,
) -> dict[str, Any]:
    recordings: list[dict[str, Any]] = []
    for path in paths:
        samples, sample_rate = load_wav(path)
        analysis = analyze_signal(
            samples,
            sample_rate,
            maximum_peaks=max(maximum_modes * 3, maximum_modes),
            minimum_frequency_hz=minimum_frequency_hz,
            maximum_frequency_hz=maximum_frequency_hz,
            spectrum_start_ms=decay_start_ms,
            spectrum_duration_ms=decay_duration_ms,
        )
        onset_sample = round(float(analysis["onset_time_s"]) * sample_rate)
        for peak in analysis["dominant_peaks"]:
            peak["decay_s"] = estimate_decay_s(
                samples,
                sample_rate,
                float(peak["frequency_hz"]),
                onset_sample=onset_sample,
                start_ms=decay_start_ms,
                duration_ms=decay_duration_ms,
            )
        recordings.append(
            {
                "file": path.name,
                "sample_rate": sample_rate,
                "analysis": analysis,
            }
        )
    modes = _cluster_peaks(recordings, maximum_modes=maximum_modes)
    return {
        "schema": "sfxrender.modal_reference.v1",
        "name": name,
        "method": "numpy_fft_peak_cluster_and_narrowband_log_decay",
        "options": {
            "minimum_frequency_hz": minimum_frequency_hz,
            "maximum_frequency_hz": maximum_frequency_hz,
            "maximum_modes": maximum_modes,
            "decay_start_ms": decay_start_ms,
            "decay_duration_ms": decay_duration_ms,
        },
        "recording_count": len(recordings),
        "modes": modes,
        "repeatability": {
            "mean_frequency_std_hz": float(
                _mean([float(mode["frequency_std_hz"]) for mode in modes])
            ),
            "mean_decay_std_s": float(_mean([float(mode["decay_std_s"]) for mode in modes])),
        },
        "recordings": [
            {
                "file": recording["file"],
                "sample_rate": recording["sample_rate"],
                "duration_s": recording["analysis"]["duration_s"],
                "onset_time_s": recording["analysis"]["onset_time_s"],
                "rms": recording["analysis"]["rms"],
                "peak": recording["analysis"]["peak"],
                "spectral_centroid_hz": recording["analysis"]["spectral_centroid_hz"],
                "event_density_hz": recording["analysis"]["event_density_hz"],
            }
            for recording in recordings
        ],
    }


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "references", nargs="+", type=Path, help="one or more uncompressed PCM WAV recordings"
    )
    parser.add_argument(
        "--name", default="reference_object", help="name for the emitted effective modal preset"
    )
    parser.add_argument(
        "--output", type=Path, help="JSON output path; omit to write JSON to stdout"
    )
    parser.add_argument("--maximum-modes", type=int, default=24)
    parser.add_argument("--minimum-frequency-hz", type=float, default=60.0)
    parser.add_argument("--maximum-frequency-hz", type=float, default=10_000.0)
    parser.add_argument("--decay-start-ms", type=float, default=100.0)
    parser.add_argument("--decay-duration-ms", type=float, default=400.0)
    args = parser.parse_args(argv)
    if args.maximum_modes < 1 or args.decay_duration_ms <= 0.0:
        parser.error("maximum-modes and decay-duration-ms must be positive")
    if args.decay_start_ms < 0.0 or args.minimum_frequency_hz <= 0.0:
        parser.error("decay-start-ms must be non-negative and minimum-frequency-hz positive")
    if args.maximum_frequency_hz <= args.minimum_frequency_hz:
        parser.error("maximum-frequency-hz must exceed minimum-frequency-hz")
    missing = [str(path) for path in args.references if not path.is_file()]
    if missing:
        parser.error("input WAV not found: " + ", ".join(missing))
    try:
        preset = fit_references(
            args.references,
            name=args.name,
            maximum_modes=args.maximum_modes,
            minimum_frequency_hz=args.minimum_frequency_hz,
            maximum_frequency_hz=args.maximum_frequency_hz,
            decay_start_ms=args.decay_start_ms,
            decay_duration_ms=args.decay_duration_ms,
        )
    except (OSError, ValueError, wave.Error) as error:
        print(f"fit_modal_reference: {error}", file=sys.stderr)
        return 2
    rendered = json.dumps(preset, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
        print(f"Wrote modal preset -> {args.output}")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

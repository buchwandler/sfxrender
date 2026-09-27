"""Compare distributions of reference and candidate WAV features without waveform alignment."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import wave
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools._reference_audio import analyze_signal, load_wav, spectral_correlation

_SCALAR_FEATURES = (
    "duration_s",
    "active_duration_s",
    "onset_time_s",
    "attack_duration_s",
    "rms",
    "peak",
    "spectral_centroid_hz",
    "spectral_rolloff_85_hz",
    "spectral_flux",
    "zero_crossing_rate",
    "event_density_hz",
)


def _analyze(paths: list[Path]) -> list[dict[str, Any]]:
    analyses: list[dict[str, Any]] = []
    for path in paths:
        samples, sample_rate = load_wav(path)
        analyses.append(
            {
                "file": path.name,
                "sample_rate": sample_rate,
                "samples": samples,
                "features": analyze_signal(samples, sample_rate),
            }
        )
    return analyses


def _median_feature(items: list[dict[str, Any]], key: str) -> float:
    return float(statistics.median(float(item["features"][key]) for item in items))


def _feature_spread(items: list[dict[str, Any]], key: str) -> dict[str, float]:
    values = sorted(float(item["features"][key]) for item in items)
    last = len(values) - 1
    return {
        "p10": values[round(last * 0.1)],
        "p90": values[round(last * 0.9)],
        "std": float(statistics.pstdev(values)),
    }


def _peak_matches(
    reference: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
) -> list[dict[str, float | None]]:
    reference_peaks = [peak for item in reference for peak in item["features"]["dominant_peaks"]]
    candidate_peaks = [peak for item in candidate for peak in item["features"]["dominant_peaks"]]
    matches: list[dict[str, float | None]] = []
    used: set[int] = set()
    for candidate_peak in candidate_peaks:
        frequency = float(candidate_peak["frequency_hz"])
        options = [
            (index, peak)
            for index, peak in enumerate(reference_peaks)
            if index not in used
            and abs(float(peak["frequency_hz"]) - frequency) <= max(12.0, frequency * 0.025)
        ]
        if not options:
            continue
        index, reference_peak = min(
            options,
            key=lambda pair: abs(float(pair[1]["frequency_hz"]) - frequency),
        )
        used.add(index)
        reference_decay_value = reference_peak["decay_s"]
        candidate_decay_value = candidate_peak["decay_s"]
        reference_decay = (
            float(reference_decay_value) if reference_decay_value is not None else None
        )
        candidate_decay = (
            float(candidate_decay_value) if candidate_decay_value is not None else None
        )
        decay_delta = (
            candidate_decay - reference_decay
            if candidate_decay is not None and reference_decay is not None
            else None
        )
        matches.append(
            {
                "reference_frequency_hz": float(reference_peak["frequency_hz"]),
                "candidate_frequency_hz": frequency,
                "frequency_delta_hz": frequency - float(reference_peak["frequency_hz"]),
                "reference_decay_s": reference_decay,
                "candidate_decay_s": candidate_decay,
                "decay_delta_s": decay_delta,
            }
        )
    return matches


def _correlation_summary(items: list[dict[str, Any]]) -> dict[str, float]:
    values: list[float] = []
    for left_index, left in enumerate(items):
        for right in items[left_index + 1 :]:
            values.append(
                spectral_correlation(
                    left["samples"],
                    left["sample_rate"],
                    right["samples"],
                    right["sample_rate"],
                )
            )
    return {
        "pair_count": len(values),
        "median_spectral_correlation": float(statistics.median(values)) if values else 0.0,
    }


def compare_sets(reference_paths: list[Path], candidate_paths: list[Path]) -> dict[str, Any]:
    reference = _analyze(reference_paths)
    candidate = _analyze(candidate_paths)
    reference_summary = {key: _median_feature(reference, key) for key in _SCALAR_FEATURES}
    candidate_summary = {key: _median_feature(candidate, key) for key in _SCALAR_FEATURES}
    reference_spread = {key: _feature_spread(reference, key) for key in _SCALAR_FEATURES}
    candidate_spread = {key: _feature_spread(candidate, key) for key in _SCALAR_FEATURES}
    trajectory_key = "spectral_centroid_trajectory_hz"
    reference_trajectory = [
        float(
            statistics.median(float(item["features"][trajectory_key][index]) for item in reference)
        )
        for index in range(5)
    ]
    candidate_trajectory = [
        float(
            statistics.median(float(item["features"][trajectory_key][index]) for item in candidate)
        )
        for index in range(5)
    ]
    deltas: dict[str, dict[str, float]] = {}
    for key in _SCALAR_FEATURES:
        difference = candidate_summary[key] - reference_summary[key]
        deltas[key] = {
            "reference_median": reference_summary[key],
            "candidate_median": candidate_summary[key],
            "absolute_delta": difference,
            "relative_delta": difference / abs(reference_summary[key])
            if reference_summary[key]
            else 0.0,
        }
    band_names = sorted(reference[0]["features"]["band_energy_fraction"])
    reference_bands = {
        name: float(
            statistics.median(
                float(item["features"]["band_energy_fraction"][name]) for item in reference
            )
        )
        for name in band_names
    }
    candidate_bands = {
        name: float(
            statistics.median(
                float(item["features"]["band_energy_fraction"][name]) for item in candidate
            )
        )
        for name in band_names
    }
    band_deltas = {name: candidate_bands[name] - reference_bands[name] for name in band_names}
    return {
        "schema": "sfxrender.reference_comparison.v1",
        "alignment": "distribution_features_no_sample_alignment",
        "recording_counts": {"reference": len(reference), "candidate": len(candidate)},
        "reference": {
            "files": [item["file"] for item in reference],
            "median_features": reference_summary,
            "feature_spread": reference_spread,
            "median_band_energy_fraction": reference_bands,
            "inter_hit_repeatability": _correlation_summary(reference),
        },
        "candidate": {
            "files": [item["file"] for item in candidate],
            "median_features": candidate_summary,
            "feature_spread": candidate_spread,
            "median_band_energy_fraction": candidate_bands,
            "inter_hit_repeatability": _correlation_summary(candidate),
        },
        "feature_deltas": deltas,
        "band_energy_fraction_deltas": band_deltas,
        "spectral_centroid_trajectory": {
            "reference_hz": reference_trajectory,
            "candidate_hz": candidate_trajectory,
            "delta_hz": [
                candidate - reference
                for reference, candidate in zip(
                    reference_trajectory, candidate_trajectory, strict=True
                )
            ],
        },
        "matched_modal_peaks": _peak_matches(reference, candidate),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference", nargs="+", required=True, type=Path, help="reference WAV recording(s)"
    )
    parser.add_argument(
        "--candidate",
        nargs="+",
        required=True,
        type=Path,
        help="rendered/candidate WAV recording(s)",
    )
    parser.add_argument(
        "--output", type=Path, help="JSON report path; omit to write JSON to stdout"
    )
    args = parser.parse_args(argv)
    paths = args.reference + args.candidate
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        parser.error("input WAV not found: " + ", ".join(missing))
    try:
        report = compare_sets(args.reference, args.candidate)
    except (OSError, ValueError, wave.Error) as error:
        print(f"compare_reference: {error}", file=sys.stderr)
        return 2
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
        print(f"Wrote feature comparison -> {args.output}")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

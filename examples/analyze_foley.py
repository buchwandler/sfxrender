"""Print and save compact spectral/temporal measurements for fixed Foley cases."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from sfxrender import SFXRenderer

ANALYSIS_CASES = (
    ("wood-boots", "sfx:footsteps.walk?surface=wood&footwear=boots&count=1&force=0.58&seed=501"),
    ("wood-shoes", "sfx:footsteps.walk?surface=wood&footwear=shoes&count=1&force=0.58&seed=502"),
    ("stone-shoes", "sfx:footsteps.walk?surface=stone&footwear=shoes&count=1&force=0.58&seed=503"),
    ("stone-heels", "sfx:footsteps.walk?surface=stone&footwear=heels&count=1&force=0.58&seed=504"),
    (
        "carpet-barefoot",
        "sfx:footsteps.walk?surface=carpet&footwear=barefoot&count=1&force=0.58&seed=505",
    ),
    (
        "gravel-boots",
        "sfx:footsteps.walk?surface=gravel&footwear=boots&count=1&force=0.58&seed=506",
    ),
)


def _rms(samples: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(samples, dtype=np.float64)))) if samples.size else 0.0


def _band_fraction(
    power: np.ndarray, frequencies: np.ndarray, low_hz: float, high_hz: float
) -> float:
    selected = power[(frequencies >= low_hz) & (frequencies < high_hz)]
    total = float(power.sum())
    return float(selected.sum() / total) if total > 0.0 else 0.0


def _measure(samples: np.ndarray, sample_rate: int) -> dict[str, float]:
    windowed = samples * np.hanning(samples.size)
    power = np.square(np.abs(np.fft.rfft(windowed)))
    frequencies = np.fft.rfftfreq(samples.size, 1.0 / sample_rate)
    total = float(power.sum())
    centroid = float(np.sum(frequencies * power) / total) if total > 0.0 else 0.0
    return {
        "duration_s": samples.size / float(sample_rate),
        "peak": float(np.max(np.abs(samples))) if samples.size else 0.0,
        "rms": _rms(samples),
        "centroid_hz": centroid,
        "fraction_80_200_hz": _band_fraction(power, frequencies, 80.0, 200.0),
        "fraction_200_1000_hz": _band_fraction(power, frequencies, 200.0, 1_000.0),
        "fraction_1_5khz": _band_fraction(power, frequencies, 1_000.0, 5_000.0),
        "dominant_bin_fraction": float(power.max() / total) if total > 0.0 else 0.0,
        "early_rms": _rms(samples[: int(0.035 * sample_rate)]),
        "mid_rms": _rms(samples[int(0.035 * sample_rate) : int(0.12 * sample_rate)]),
        "tail_rms": _rms(samples[int(0.12 * sample_rate) : int(0.22 * sample_rate)]),
    }


def main(output_dir: str | Path | None = None) -> None:
    """Write deterministic analysis rows beneath ``output_dir`` or example-artifacts/."""
    destination = (
        Path(output_dir)
        if output_dir is not None
        else Path(__file__).resolve().parents[1] / "example-artifacts"
    )
    destination.mkdir(parents=True, exist_ok=True)
    renderer = SFXRenderer(sample_rate=24_000)
    keys = tuple(_measure(np.zeros(1, dtype=np.float32), renderer.sample_rate))
    lines = ["case " + " ".join(keys)]
    for name, uri in ANALYSIS_CASES:
        sound = renderer.render_uri(uri)
        metrics = _measure(sound.samples, sound.sample_rate)
        row = " ".join(f"{metrics[key]:.6f}" for key in keys)
        lines.append(f"{name} {row}")
    report = "\n".join(lines) + "\n"
    output_path = destination / "foley-analysis.txt"
    output_path.write_text(report, encoding="utf-8")
    print(report, end="")
    print(f"Wrote analysis -> {output_path}")


if __name__ == "__main__":
    main()

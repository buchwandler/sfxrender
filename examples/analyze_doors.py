"""Write NumPy-only signal and generated-model metrics for fixed door cases."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from sfxrender import SFXRenderer
from sfxrender._doors import generate_door_model

ANALYSIS_CASES = (
    (
        "open-wood-slow",
        "sfx:door.open?material=wood&speed=slow&creak=0.75&seed=501",
        "wood",
        501,
        "open",
        2.25,
    ),
    (
        "open-wood-normal",
        "sfx:door.open?material=wood&speed=normal&creak=0.5&seed=502",
        "wood",
        502,
        "open",
        1.42,
    ),
    (
        "open-wood-fast",
        "sfx:door.open?material=wood&speed=fast&creak=0.5&seed=503",
        "wood",
        503,
        "open",
        0.78,
    ),
    (
        "open-metal-normal",
        "sfx:door.open?material=metal&speed=normal&creak=0.55&seed=504",
        "metal",
        504,
        "open",
        1.42,
    ),
    (
        "close-wood-slow",
        "sfx:door.close?material=wood&speed=slow&force=0.55&creak=0.25&seed=501",
        "wood",
        501,
        "close",
        0.82,
    ),
    (
        "close-wood-normal",
        "sfx:door.close?material=wood&speed=normal&force=0.65&creak=0.25&seed=505",
        "wood",
        505,
        "close",
        0.46,
    ),
    (
        "close-wood-fast-hard",
        "sfx:door.close?material=wood&speed=fast&force=0.9&creak=0.2&seed=506",
        "wood",
        506,
        "close",
        0.23,
    ),
    (
        "close-metal-normal",
        "sfx:door.close?material=metal&speed=normal&force=0.7&creak=0.3&seed=507",
        "metal",
        507,
        "close",
        0.46,
    ),
)


_METRIC_KEYS = (
    "duration_s",
    "peak",
    "rms",
    "spectral_centroid_hz",
    "spectral_flatness",
    "fraction_40_250_hz",
    "fraction_250_1200_hz",
    "fraction_1200_5000_hz",
    "fraction_5000_8000_hz",
    "early_rms",
    "motion_rms",
    "terminal_rms",
    "frame_rms_p25",
    "frame_rms_p90",
    "intermittency_ratio",
    "panel_mode_count",
    "panel_mode_min_hz",
    "panel_mode_max_hz",
    "panel_mean_decay_s",
    "frame_mode_count",
    "hinge_region_count",
    "hinge_f0_min_hz",
    "hinge_f0_max_hz",
)


def _rms(samples: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(samples, dtype=np.float64)))) if samples.size else 0.0


def _band_fraction(power: np.ndarray, frequencies: np.ndarray, low: float, high: float) -> float:
    selected = power[(frequencies >= low) & (frequencies < high)]
    total = float(power.sum())
    return float(selected.sum() / total) if total > 0.0 else 0.0


def _signal_metrics(
    samples: np.ndarray, sample_rate: int, action: str, motion_duration: float
) -> dict[str, float]:
    windowed = samples * np.hanning(samples.size)
    power = np.square(np.abs(np.fft.rfft(windowed)))
    frequencies = np.fft.rfftfreq(samples.size, 1.0 / sample_rate)
    total = float(power.sum())
    centroid = float(np.sum(frequencies * power) / total) if total > 0.0 else 0.0
    positive_power = power[power > 0.0]
    flatness = (
        float(np.exp(np.mean(np.log(positive_power))) / np.mean(positive_power))
        if positive_power.size
        else 0.0
    )
    if action == "open":
        motion_start = int(0.16 * sample_rate)
        motion_end = int(max(0.17, motion_duration - 0.13) * sample_rate)
    else:
        motion_start = int(0.07 * sample_rate)
        motion_end = int(max(0.08, motion_duration - 0.035) * sample_rate)
    motion_segment = samples[motion_start:motion_end]
    terminal = samples[int(motion_duration * sample_rate) :]
    frame_size = max(1, round(0.05 * sample_rate))
    frame_count = samples.size // frame_size
    if frame_count:
        frames = samples[: frame_count * frame_size].reshape(frame_count, frame_size)
        frame_rms = np.sqrt(np.mean(np.square(frames.astype(np.float64)), axis=1))
        p25 = float(np.percentile(frame_rms, 25))
        p90 = float(np.percentile(frame_rms, 90))
    else:
        p25 = 0.0
        p90 = 0.0
    return {
        "duration_s": samples.size / float(sample_rate),
        "peak": float(np.max(np.abs(samples))) if samples.size else 0.0,
        "rms": _rms(samples),
        "spectral_centroid_hz": centroid,
        "spectral_flatness": flatness,
        "fraction_40_250_hz": _band_fraction(power, frequencies, 40.0, 250.0),
        "fraction_250_1200_hz": _band_fraction(power, frequencies, 250.0, 1_200.0),
        "fraction_1200_5000_hz": _band_fraction(power, frequencies, 1_200.0, 5_000.0),
        "fraction_5000_8000_hz": _band_fraction(power, frequencies, 5_000.0, 8_000.0),
        "early_rms": _rms(samples[: int(0.12 * sample_rate)]),
        "motion_rms": _rms(motion_segment),
        "terminal_rms": _rms(terminal),
        "frame_rms_p25": p25,
        "frame_rms_p90": p90,
        "intermittency_ratio": p90 / max(p25, 1e-12),
    }


def _body_metrics(material: str, seed: int, sample_rate: int) -> dict[str, float]:
    model = generate_door_model(material=material, seed=seed, sample_rate=sample_rate)
    frequencies = [mode.frequency_hz for mode in model.panel.modes]
    decays = [mode.decay_s * model.panel.damping_scale for mode in model.panel.modes]
    return {
        "panel_mode_count": float(len(model.panel.modes)),
        "panel_mode_min_hz": min(frequencies, default=0.0),
        "panel_mode_max_hz": max(frequencies, default=0.0),
        "panel_mean_decay_s": float(np.mean(decays)) if decays else 0.0,
        "frame_mode_count": float(len(model.frame.modes)),
        "hinge_region_count": float(len(model.hinge_regions)),
        "hinge_f0_min_hz": model.hinge_friction.f0_hz[0],
        "hinge_f0_max_hz": model.hinge_friction.f0_hz[1],
    }


def main(output_dir: str | Path | None = None) -> None:
    """Print and save reproducible metrics under ``output_dir`` or example-artifacts/."""
    destination = (
        Path(output_dir)
        if output_dir is not None
        else Path(__file__).resolve().parents[1] / "example-artifacts"
    )
    destination.mkdir(parents=True, exist_ok=True)
    renderer = SFXRenderer(sample_rate=24_000)
    lines = ["case " + " ".join(_METRIC_KEYS)]
    for name, uri, material, seed, action, motion_duration in ANALYSIS_CASES:
        sound = renderer.render_uri(uri)
        metrics = _signal_metrics(sound.samples, sound.sample_rate, action, motion_duration)
        metrics.update(_body_metrics(material, seed, sound.sample_rate))
        values = " ".join(f"{metrics[key]:.6f}" for key in _METRIC_KEYS)
        lines.append(f"{name} {values}")
    report = "\n".join(lines) + "\n"
    output_path = destination / "door-analysis.txt"
    output_path.write_text(report, encoding="utf-8")
    print(report, end="")
    print(f"Wrote door analysis -> {output_path}")


if __name__ == "__main__":
    main()

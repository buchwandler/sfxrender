"""Write deterministic transient and spectrum metrics for keyboard key classes."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from sfxrender._keyboard import KeyEvent, _generate_typing_events, _render_keyboard_events

SAMPLE_RATE = 24_000
SEED = 123
_PRESS_TIME_S = 0.05
_RELEASE_TIME_S = 0.24
_RENDER_DURATION_S = 0.30

_METRIC_KEYS = (
    "duration_s",
    "peak",
    "rms",
    "onset_time_s",
    "attack_duration_s",
    "active_duration_s",
    "spectral_centroid_hz",
    "spectral_flatness",
    "zero_crossing_rate",
    "dominant_peak_hz",
    "dominant_peak_relative_amplitude",
    "dominant_bin_fraction",
    "fraction_80_250_hz",
    "fraction_250_1200_hz",
    "fraction_1200_5000_hz",
    "fraction_5000_9000_hz",
    "early_rms",
    "mid_rms",
    "tail_rms",
    "tail_to_early_ratio",
)


def _rms(samples: np.ndarray) -> float:
    if not samples.size:
        return 0.0
    return float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))


def _analysis_onset(samples: np.ndarray, sample_rate: int) -> int:
    frame_size = max(1, round(0.005 * sample_rate))
    frame_count = (samples.size + frame_size - 1) // frame_size
    if frame_count == 0:
        return 0
    padded = np.pad(samples.astype(np.float64), (0, frame_count * frame_size - samples.size))
    frames = padded.reshape(frame_count, frame_size)
    frame_rms = np.sqrt(np.mean(np.square(frames), axis=1))
    peak = float(np.max(frame_rms))
    active = np.flatnonzero(frame_rms >= max(peak * 0.08, 1e-8))
    return int(active[0] * frame_size) if active.size else 0


def _band_fraction(power: np.ndarray, frequencies: np.ndarray, low_hz: float, high_hz: float) -> float:
    total = float(power.sum())
    if total <= 0.0:
        return 0.0
    selected = power[(frequencies >= low_hz) & (frequencies < high_hz)]
    return float(selected.sum() / total)


def _measure(
    samples: np.ndarray,
    sample_rate: int,
    *,
    onset_sample: int,
    release_sample: int,
) -> dict[str, float]:
    signal = np.asarray(samples, dtype=np.float64)
    if signal.ndim != 1 or not np.all(np.isfinite(signal)):
        raise ValueError("keyboard analysis requires finite mono audio")

    onset = _analysis_onset(signal, sample_rate)
    onset_s = onset / sample_rate
    frame_size = max(1, round(0.005 * sample_rate))
    frame_count = (signal.size + frame_size - 1) // frame_size
    padded = np.pad(signal, (0, frame_count * frame_size - signal.size))
    frames = padded.reshape(frame_count, frame_size)
    frame_rms = np.sqrt(np.mean(np.square(frames), axis=1))
    envelope_peak = float(np.max(frame_rms)) if frame_rms.size else 0.0
    active_frames = np.flatnonzero(frame_rms >= max(envelope_peak * 0.08, 1e-8))
    peak_frame = int(np.argmax(frame_rms)) if frame_rms.size else 0
    attack_duration = max(0.0, (peak_frame * frame_size - onset) / sample_rate)
    active_duration = (
        max(0.0, ((int(active_frames[-1]) - int(active_frames[0]) + 1) * frame_size) / sample_rate)
        if active_frames.size
        else 0.0
    )

    windowed = signal * np.hanning(signal.size) if signal.size else signal
    power = np.square(np.abs(np.fft.rfft(windowed)))
    frequencies = np.fft.rfftfreq(signal.size, d=1.0 / sample_rate) if signal.size else np.zeros(0)
    total_power = float(power.sum())
    centroid = float(np.dot(frequencies, power) / total_power) if total_power > 0.0 else 0.0
    positive_power = power[power > 0.0]
    flatness = (
        float(np.exp(np.mean(np.log(positive_power))) / np.mean(positive_power))
        if positive_power.size
        else 0.0
    )
    non_dc = np.arange(power.size) > 0
    peak_bin = int(np.argmax(np.where(non_dc, power, 0.0))) if power.size > 1 else 0
    magnitude = np.sqrt(power)
    local_peaks = np.flatnonzero(
        (magnitude[1:-1] >= magnitude[:-2]) & (magnitude[1:-1] > magnitude[2:])
    ) + 1 if magnitude.size >= 3 else np.zeros(0, dtype=int)
    dominant_peak_bin = (
        int(local_peaks[np.argmax(magnitude[local_peaks])]) if local_peaks.size else peak_bin
    )
    dominant_peak = float(frequencies[dominant_peak_bin]) if frequencies.size else 0.0
    dominant_peak_relative = (
        float(magnitude[dominant_peak_bin] / max(float(magnitude.sum()), 1e-15))
        if magnitude.size
        else 0.0
    )

    early_start = max(0, onset_sample)
    early_end = min(signal.size, early_start + round(0.025 * sample_rate))
    mid_start = min(signal.size, early_start + round(0.025 * sample_rate))
    mid_end = min(signal.size, early_start + round(0.080 * sample_rate), release_sample)
    tail_start = min(signal.size, early_start + round(0.080 * sample_rate))
    tail_end = min(signal.size, early_start + round(0.140 * sample_rate), release_sample)
    early_rms = _rms(signal[early_start:early_end])
    mid_rms = _rms(signal[mid_start:mid_end])
    tail_rms = _rms(signal[tail_start:tail_end])

    return {
        "duration_s": signal.size / float(sample_rate),
        "peak": float(np.max(np.abs(signal))) if signal.size else 0.0,
        "rms": _rms(signal),
        "onset_time_s": onset_s,
        "attack_duration_s": attack_duration,
        "active_duration_s": active_duration,
        "spectral_centroid_hz": centroid,
        "spectral_flatness": flatness,
        "zero_crossing_rate": float(np.mean(signal[1:] * signal[:-1] < 0.0))
        if signal.size > 1
        else 0.0,
        "dominant_peak_hz": dominant_peak,
        "dominant_peak_relative_amplitude": dominant_peak_relative,
        "dominant_bin_fraction": float(power[peak_bin] / total_power) if total_power > 0.0 else 0.0,
        "fraction_80_250_hz": _band_fraction(power, frequencies, 80.0, 250.0),
        "fraction_250_1200_hz": _band_fraction(power, frequencies, 250.0, 1_200.0),
        "fraction_1200_5000_hz": _band_fraction(power, frequencies, 1_200.0, 5_000.0),
        "fraction_5000_9000_hz": _band_fraction(power, frequencies, 5_000.0, 9_000.0),
        "early_rms": early_rms,
        "mid_rms": mid_rms,
        "tail_rms": tail_rms,
        "tail_to_early_ratio": tail_rms / max(early_rms, 1e-12),
    }


def _isolated_case(name: str, key_class: str, variant: int, velocity: float) -> tuple[str, np.ndarray, int, int]:
    event = KeyEvent(
        _PRESS_TIME_S,
        _RELEASE_TIME_S,
        key_class,
        variant,
        velocity,
        0.5,
    )
    samples = _render_keyboard_events(
        (event,), duration=_RENDER_DURATION_S, sample_rate=SAMPLE_RATE, seed=SEED
    )
    return (
        name,
        samples,
        round(_PRESS_TIME_S * SAMPLE_RATE),
        round(_RELEASE_TIME_S * SAMPLE_RATE),
    )


def _cases() -> tuple[tuple[str, np.ndarray, int, int], ...]:
    cases = [
        _isolated_case("normal-light-v0", "normal", 0, 0.115),
        _isolated_case("normal-firm-v0", "normal", 0, 0.205),
    ]
    cases.extend(
        _isolated_case(f"normal-light-v{variant}", "normal", variant, 0.115)
        for variant in range(1, 5)
    )
    cases.extend(
        _isolated_case(f"{key_class}-light", key_class, 0, 0.115)
        for key_class in ("space", "enter", "backspace")
    )
    release_name, release_samples, _, release_index = _isolated_case(
        "normal-release-focused", "normal", 0, 0.115
    )
    release_samples = release_samples[release_index:]
    cases.append((release_name, release_samples, 0, release_samples.size))
    for speed, force in (("slow", "light"), ("steady", "light"), ("fast", "light"), ("fast", "firm")):
        events = _generate_typing_events(duration=3.0, speed=speed, force=force, seed=SEED)
        samples = _render_keyboard_events(
            events, duration=3.0, sample_rate=SAMPLE_RATE, seed=SEED, speed=speed
        )
        if events:
            onset_sample = round(events[0].press_time_s * SAMPLE_RATE)
            release_sample = round(events[0].release_time_s * SAMPLE_RATE)
        else:
            onset_sample = 0
            release_sample = samples.size
        cases.append((f"typing-{speed}-{force}", samples, onset_sample, release_sample))
    return tuple(cases)


def main(output_dir: str | Path | None = None) -> None:
    """Write deterministic analysis metrics below the requested directory."""
    destination = (
        Path(output_dir)
        if output_dir is not None
        else Path(__file__).resolve().parents[1] / "example-artifacts"
    )
    destination.mkdir(parents=True, exist_ok=True)
    lines = ["case " + " ".join(_METRIC_KEYS)]
    for name, samples, onset_sample, release_sample in _cases():
        metrics = _measure(
            samples,
            SAMPLE_RATE,
            onset_sample=onset_sample,
            release_sample=release_sample,
        )
        lines.append(name + " " + " ".join(f"{metrics[key]:.6f}" for key in _METRIC_KEYS))
    report = "\n".join(lines) + "\n"
    output_path = destination / "keyboard-analysis.txt"
    output_path.write_text(report, encoding="utf-8")
    print(report, end="")
    print(f"Wrote keyboard analysis -> {output_path}")


if __name__ == "__main__":
    main()

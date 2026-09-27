"""NumPy-only WAV loading and signal measurements for developer calibration tools."""

from __future__ import annotations

import math
import wave
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


def load_wav(path: str | Path) -> tuple[FloatArray, int]:
    """Read integer PCM WAV, downmixing channels to a finite mono float64 signal."""
    source = Path(path)
    with wave.open(str(source), "rb") as stream:
        channels = stream.getnchannels()
        sample_rate = stream.getframerate()
        sample_width = stream.getsampwidth()
        frame_count = stream.getnframes()
        if stream.getcomptype() != "NONE":
            raise ValueError(f"{source}: only uncompressed PCM WAV is supported")
        if channels < 1 or sample_rate < 1 or frame_count < 1:
            raise ValueError(f"{source}: WAV must contain audio frames and a valid format")
        raw = stream.readframes(frame_count)
    if sample_width == 1:
        pcm = np.frombuffer(raw, dtype=np.uint8).astype(np.float64) - 128.0
        pcm /= 128.0
    elif sample_width == 2:
        pcm = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32_768.0
    elif sample_width == 3:
        octets = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int64)
        unsigned = octets[:, 0] | (octets[:, 1] << 8) | (octets[:, 2] << 16)
        signed = (unsigned ^ 0x800000) - 0x800000
        pcm = signed.astype(np.float64) / 8_388_608.0
    elif sample_width == 4:
        pcm = np.frombuffer(raw, dtype="<i4").astype(np.float64) / 2_147_483_648.0
    else:
        raise ValueError(f"{source}: unsupported PCM sample width {sample_width} bytes")
    if pcm.size != frame_count * channels:
        raise ValueError(f"{source}: truncated PCM data")
    mono = pcm.reshape(-1, channels).mean(axis=1) if channels > 1 else pcm
    if not np.all(np.isfinite(mono)):
        raise ValueError(f"{source}: decoded samples are non-finite")
    return np.asarray(mono, dtype=np.float64), int(sample_rate)


def _frame_envelope(samples: FloatArray, sample_rate: int) -> tuple[FloatArray, FloatArray]:
    frame_size = max(16, round(sample_rate * 0.005))
    hop = max(1, round(sample_rate * 0.001))
    starts = np.arange(0, max(1, samples.size), hop, dtype=np.int64)
    rms = np.empty(starts.size, dtype=np.float64)
    for index, start in enumerate(starts):
        frame = samples[start : min(samples.size, start + frame_size)]
        rms[index] = math.sqrt(float(np.mean(frame * frame))) if frame.size else 0.0
    return starts.astype(np.float64) / sample_rate, rms


def _find_spectral_peaks(
    samples: FloatArray,
    sample_rate: int,
    *,
    minimum_hz: float,
    maximum_hz: float,
    count: int,
    minimum_separation_hz: float = 12.0,
    minimum_peak_db: float = -38.0,
) -> list[dict[str, Any]]:
    if samples.size < 8 or count <= 0:
        return []
    windowed = samples * np.hanning(samples.size)
    requested_nfft = max(16_384, samples.size * 4)
    nfft = min(524_288, 1 << math.ceil(math.log2(requested_nfft)))
    magnitude = np.abs(np.fft.rfft(windowed, n=nfft))
    frequencies = np.fft.rfftfreq(nfft, d=1.0 / sample_rate)
    band = np.flatnonzero(
        (frequencies >= minimum_hz) & (frequencies <= min(maximum_hz, sample_rate * 0.49))
    )
    if band.size < 3:
        return []
    floor = float(np.max(magnitude[band])) * 10.0 ** (minimum_peak_db / 20.0)
    candidate_bins = band[1:-1]
    local_maxima = candidate_bins[
        (magnitude[candidate_bins] >= magnitude[candidate_bins - 1])
        & (magnitude[candidate_bins] > magnitude[candidate_bins + 1])
        & (magnitude[candidate_bins] >= floor)
    ]
    ranked = sorted(local_maxima.tolist(), key=lambda index: float(magnitude[index]), reverse=True)
    selected: list[dict[str, Any]] = []
    for index in ranked:
        left = math.log(max(float(magnitude[index - 1]), 1e-15))
        center = math.log(max(float(magnitude[index]), 1e-15))
        right = math.log(max(float(magnitude[index + 1]), 1e-15))
        denominator = left - 2.0 * center + right
        offset = 0.5 * (left - right) / denominator if abs(denominator) > 1e-12 else 0.0
        offset = float(np.clip(offset, -0.5, 0.5))
        frequency = (index + offset) * sample_rate / nfft
        if any(abs(frequency - peak["frequency_hz"]) < minimum_separation_hz for peak in selected):
            continue
        relative = float(magnitude[index] / max(float(np.max(magnitude[band])), 1e-15))
        selected.append(
            {
                "frequency_hz": float(frequency),
                "relative_amplitude": relative,
                "level_db": float(20.0 * math.log10(max(relative, 1e-15))),
            }
        )
        if len(selected) >= count:
            break
    return sorted(selected, key=lambda peak: peak["frequency_hz"])


def estimate_decay_s(
    samples: FloatArray,
    sample_rate: int,
    frequency_hz: float,
    *,
    onset_sample: int,
    start_ms: float = 100.0,
    duration_ms: float = 400.0,
) -> float | None:
    """Fit a log-amplitude slope using Hann-windowed narrowband demodulation."""
    start = max(0, onset_sample + round(start_ms * sample_rate / 1_000.0))
    end = min(samples.size, start + round(duration_ms * sample_rate / 1_000.0))
    segment = samples[start:end]
    block_size = max(32, round(sample_rate * 0.012))
    hop = max(1, block_size // 2)
    if segment.size < block_size * 3 or frequency_hz <= 0.0:
        return None
    window = np.hanning(block_size)
    denominator = max(float(window.sum()), 1e-12)
    values: list[float] = []
    times: list[float] = []
    angular = 2.0 * math.pi * frequency_hz / sample_rate
    for offset in range(0, segment.size - block_size + 1, hop):
        frame = segment[offset : offset + block_size]
        phase = np.exp(-1j * angular * np.arange(block_size, dtype=np.float64))
        amplitude = 2.0 * abs(np.sum(frame * window * phase)) / denominator
        values.append(max(float(amplitude), 1e-15))
        times.append((offset + block_size / 2.0) / sample_rate)
    envelope = np.asarray(values, dtype=np.float64)
    time_axis = np.asarray(times, dtype=np.float64)
    valid = envelope >= float(np.max(envelope)) * 0.025
    if int(np.count_nonzero(valid)) < 4:
        return None
    slope = float(np.polyfit(time_axis[valid], np.log(envelope[valid]), 1)[0])
    if not math.isfinite(slope) or slope >= -0.01:
        return None
    return float(np.clip(-1.0 / slope, 0.001, 20.0))


def _spectrum_metrics(
    samples: FloatArray, sample_rate: int
) -> tuple[float, float, dict[str, float]]:
    if samples.size == 0:
        return 0.0, 0.0, {"0_250_hz": 0.0, "250_1000_hz": 0.0, "1k_5k_hz": 0.0, "5k_12k_hz": 0.0}
    windowed = samples * np.hanning(samples.size)
    power = np.square(np.abs(np.fft.rfft(windowed)))
    frequencies = np.fft.rfftfreq(samples.size, d=1.0 / sample_rate)
    total = float(power.sum())
    if total <= 0.0:
        return 0.0, 0.0, {"0_250_hz": 0.0, "250_1000_hz": 0.0, "1k_5k_hz": 0.0, "5k_12k_hz": 0.0}
    centroid = float(np.sum(frequencies * power) / total)
    cumulative = np.cumsum(power)
    rolloff_index = int(np.searchsorted(cumulative, 0.85 * total, side="left"))
    rolloff = float(frequencies[min(rolloff_index, frequencies.size - 1)])
    bands: dict[str, float] = {}
    for label, low, high in (
        ("0_250_hz", 0.0, 250.0),
        ("250_1000_hz", 250.0, 1_000.0),
        ("1k_5k_hz", 1_000.0, 5_000.0),
        ("5k_12k_hz", 5_000.0, 12_000.0),
    ):
        bands[label] = float(
            power[(frequencies >= low) & (frequencies < min(high, sample_rate / 2))].sum() / total
        )
    return centroid, rolloff, bands


def analyze_signal(
    samples: npt.ArrayLike,
    sample_rate: int,
    *,
    maximum_peaks: int = 16,
    minimum_frequency_hz: float = 60.0,
    maximum_frequency_hz: float = 8_000.0,
    spectrum_start_ms: float = 0.0,
    spectrum_duration_ms: float = 500.0,
) -> dict[str, Any]:
    """Measure temporal, spectral, peak, and event features for mono audio."""
    signal = np.asarray(samples, dtype=np.float64)
    if signal.ndim != 1 or not np.all(np.isfinite(signal)):
        raise ValueError("samples must be a finite mono signal")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if (
        maximum_peaks < 0
        or minimum_frequency_hz <= 0.0
        or maximum_frequency_hz <= minimum_frequency_hz
    ):
        raise ValueError("peak count and frequency range are invalid")
    size = signal.size
    peak = float(np.max(np.abs(signal))) if size else 0.0
    rms = math.sqrt(float(np.mean(signal * signal))) if size else 0.0
    frame_times, envelope = (
        _frame_envelope(signal, sample_rate) if size else (np.zeros(0), np.zeros(0))
    )
    envelope_peak = float(np.max(envelope)) if envelope.size else 0.0
    active = np.flatnonzero(envelope >= max(envelope_peak * 0.08, 1e-7))
    onset_sample = round(frame_times[int(active[0])] * sample_rate) if active.size else 0
    onset_time = onset_sample / sample_rate
    frame_width_s = max(16.0 / sample_rate, 0.005)
    active_duration = (
        max(0.0, float(frame_times[int(active[-1])] - onset_time)) + frame_width_s
        if active.size
        else 0.0
    )
    attack_frames = (
        np.flatnonzero(envelope >= envelope_peak * 0.90)
        if envelope_peak > 0.0
        else np.zeros(0, dtype=int)
    )
    attack_duration = (
        max(0.0, float(frame_times[int(attack_frames[0])] - onset_time))
        if attack_frames.size
        else 0.0
    )
    refractory = (
        max(1, round(0.045 / max(float(np.mean(np.diff(frame_times))), 1e-12)))
        if frame_times.size > 1
        else 1
    )
    event_threshold = envelope_peak * 0.28
    crossings = (
        np.flatnonzero((envelope[1:] >= event_threshold) & (envelope[:-1] < event_threshold)) + 1
    )
    event_peaks: list[int] = []
    for crossing in crossings.tolist():
        if not event_peaks or crossing - event_peaks[-1] >= refractory:
            event_peaks.append(crossing)
    centroid, rolloff, band_energy = _spectrum_metrics(signal, sample_rate)
    start = min(size, onset_sample + round(spectrum_start_ms * sample_rate / 1_000.0))
    spectrum_size = max(8, round(spectrum_duration_ms * sample_rate / 1_000.0))
    spectral_segment = signal[start : min(size, start + spectrum_size)]
    peaks = _find_spectral_peaks(
        spectral_segment,
        sample_rate,
        minimum_hz=minimum_frequency_hz,
        maximum_hz=maximum_frequency_hz,
        count=maximum_peaks,
    )
    for modal_peak in peaks:
        modal_peak["decay_s"] = estimate_decay_s(
            signal,
            sample_rate,
            float(modal_peak["frequency_hz"]),
            onset_sample=onset_sample,
        )
    remaining_samples = max(0, size - onset_sample)
    trajectory_window = max(8, min(round(0.1 * sample_rate), max(8, remaining_samples // 5)))
    trajectory: list[float] = []
    for fraction in (0.0, 0.1, 0.25, 0.5, 0.9):
        left = min(size, onset_sample + round(fraction * remaining_samples))
        right = min(size, left + trajectory_window)
        segment_centroid, _, _ = _spectrum_metrics(signal[left:right], sample_rate)
        trajectory.append(segment_centroid)
    zcr = float(np.mean(signal[1:] * signal[:-1] < 0.0)) if size > 1 else 0.0
    positive_envelope = envelope[envelope > 0.0]
    spectral_flux = 0.0
    if size >= 64:
        frame_size = max(64, round(sample_rate * 0.032))
        hop = max(1, frame_size // 2)
        previous: FloatArray | None = None
        flux_values: list[float] = []
        for offset in range(0, max(1, size - frame_size + 1), hop):
            frame = signal[offset : min(size, offset + frame_size)]
            if frame.size < frame_size:
                frame = np.pad(frame, (0, frame_size - frame.size))
            spectrum = np.abs(np.fft.rfft(frame * np.hanning(frame_size)))
            normalized = spectrum / max(float(np.linalg.norm(spectrum)), 1e-12)
            if previous is not None:
                flux_values.append(float(np.sum(np.maximum(normalized - previous, 0.0) ** 2)))
            previous = normalized
        spectral_flux = float(np.mean(flux_values)) if flux_values else 0.0
    return {
        "sample_rate": int(sample_rate),
        "duration_s": size / float(sample_rate),
        "active_duration_s": active_duration,
        "onset_time_s": onset_time,
        "attack_duration_s": attack_duration,
        "rms": rms,
        "peak": peak,
        "spectral_centroid_hz": centroid,
        "spectral_rolloff_85_hz": rolloff,
        "band_energy_fraction": band_energy,
        "dominant_peaks": peaks,
        "spectral_centroid_trajectory_hz": trajectory,
        "spectral_flux": spectral_flux,
        "zero_crossing_rate": zcr,
        "event_density_hz": len(event_peaks) / max(size / sample_rate, 1.0 / sample_rate),
        "envelope_peak_rms": envelope_peak,
        "envelope_rms_p10": float(np.percentile(positive_envelope, 10))
        if positive_envelope.size
        else 0.0,
        "envelope_rms_p90": float(np.percentile(positive_envelope, 90))
        if positive_envelope.size
        else 0.0,
    }


def spectral_correlation(
    left: npt.ArrayLike,
    left_rate: int,
    right: npt.ArrayLike,
    right_rate: int,
    *,
    bins: int = 2_048,
) -> float:
    """Return correlation between log-power spectra on a common frequency grid."""
    left_signal = np.asarray(left, dtype=np.float64)
    right_signal = np.asarray(right, dtype=np.float64)
    upper = min(left_rate, right_rate) * 0.49
    grid = np.linspace(0.0, upper, bins, dtype=np.float64)
    vectors: list[FloatArray] = []
    for signal, rate in ((left_signal, left_rate), (right_signal, right_rate)):
        if signal.size == 0:
            return 0.0
        nfft = max(2_048, min(131_072, 1 << math.ceil(math.log2(signal.size))))
        magnitude = np.abs(np.fft.rfft(signal * np.hanning(signal.size), n=nfft))
        frequencies = np.fft.rfftfreq(nfft, d=1.0 / rate)
        interpolated = np.interp(grid, frequencies, magnitude)
        logarithmic = np.log1p(interpolated / max(float(np.max(interpolated)), 1e-12))
        norm = float(np.linalg.norm(logarithmic))
        vectors.append(logarithmic / max(norm, 1e-12))
    correlation = float(np.corrcoef(vectors[0], vectors[1])[0, 1])
    return float(np.clip(correlation if math.isfinite(correlation) else 0.0, -1.0, 1.0))

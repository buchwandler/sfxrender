"""Minimal WAV writer with no dependency beyond NumPy."""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import numpy.typing as npt


def write_wav(path: Path, samples: npt.NDArray[np.float32], sample_rate: int) -> None:
    if samples.ndim != 1:
        raise ValueError("MVP WAV writer accepts mono audio only")
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(samples, -1.0, 1.0)
    pcm = np.rint(clipped * 32767.0).astype("<i2", copy=False)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(pcm.tobytes())

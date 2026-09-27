"""Public value types."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

FloatAudio = npt.NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class SfxSpec:
    """Parsed semantic SFX request."""

    effect: str
    parameters: Mapping[str, str]

    @property
    def seed(self) -> int | None:
        value = self.parameters.get("seed")
        return None if value is None else int(value)


@dataclass(frozen=True, slots=True)
class RenderContext:
    """Runtime settings shared with effect renderers."""

    sample_rate: int = 24_000


@dataclass(frozen=True, slots=True)
class RenderedSound:
    """Non-empty, finite, normalized mono float32 PCM and its render metadata."""

    samples: FloatAudio
    sample_rate: int
    spec: SfxSpec

    @property
    def duration(self) -> float:
        return float(self.samples.size) / float(self.sample_rate)

    def write_wav(self, path: str | Path) -> Path:
        from .wav import write_wav

        output = Path(path)
        write_wav(output, self.samples, self.sample_rate)
        return output

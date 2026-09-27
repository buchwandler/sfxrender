"""Persistent exact-step damped modal oscillator bank."""

from __future__ import annotations

import math

import numpy as np

from ..types import FloatAudio
from .modes import ModeSet, filter_audio_modes


class ModalResonatorBank:
    """Stateful force-driven modal bank using stable continuous-state transitions.

    The input is held constant over each audio sample. Each mode's under/over-
    damped state transition is computed analytically, avoiding forward Euler and
    preserving oscillator state between successive calls to :meth:`process`.
    """

    def __init__(
        self,
        modes: ModeSet,
        sample_rate: int,
        *,
        nyquist_fraction: float = 0.46,
    ) -> None:
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        self.sample_rate = int(sample_rate)
        self.modes = filter_audio_modes(
            modes, sample_rate=sample_rate, nyquist_fraction=nyquist_fraction
        ).modes
        count = len(self.modes)
        self._q = np.zeros(count, dtype=np.float64)
        self._v = np.zeros(count, dtype=np.float64)
        self._a00 = np.zeros(count, dtype=np.float64)
        self._a01 = np.zeros(count, dtype=np.float64)
        self._a10 = np.zeros(count, dtype=np.float64)
        self._a11 = np.zeros(count, dtype=np.float64)
        self._bq = np.zeros(count, dtype=np.float64)
        self._bv = np.zeros(count, dtype=np.float64)
        self._output_gain = np.zeros(count, dtype=np.float64)
        dt = 1.0 / self.sample_rate
        for index, mode in enumerate(self.modes):
            omega = 2.0 * math.pi * mode.frequency_hz
            sigma = 1.0 / mode.decay_s
            discriminant = omega * omega - sigma * sigma
            if discriminant > omega * omega * 1e-14:
                wd = math.sqrt(discriminant)
                decay = math.exp(-sigma * dt)
                cosine = math.cos(wd * dt)
                sine_over_wd = math.sin(wd * dt) / wd
                a00 = decay * (cosine + sigma * sine_over_wd)
                a01 = decay * sine_over_wd
                a10 = -decay * omega * omega * sine_over_wd
                a11 = decay * (cosine - sigma * sine_over_wd)
            elif discriminant < -omega * omega * 1e-14:
                root = math.sqrt(-discriminant)
                # Compute real exponentials separately to avoid cosh overflow.
                slow = math.exp((-sigma + root) * dt)
                fast = math.exp((-sigma - root) * dt)
                cosine_h = 0.5 * (slow + fast)
                sine_h_over_root = (slow - fast) / (2.0 * root)
                a00 = cosine_h + sigma * sine_h_over_root
                a01 = sine_h_over_root
                a10 = -omega * omega * sine_h_over_root
                a11 = cosine_h - sigma * sine_h_over_root
            else:
                decay = math.exp(-sigma * dt)
                a00 = decay * (1.0 + sigma * dt)
                a01 = decay * dt
                a10 = -decay * omega * omega * dt
                a11 = decay * (1.0 - sigma * dt)
            force_to_acceleration = mode.input_gain / mode.modal_mass_kg
            equilibrium_per_force = force_to_acceleration / (omega * omega)
            self._a00[index] = a00
            self._a01[index] = a01
            self._a10[index] = a10
            self._a11[index] = a11
            self._bq[index] = (1.0 - a00) * equilibrium_per_force
            self._bv[index] = -a10 * equilibrium_per_force
            self._output_gain[index] = mode.radiation_gain

    def reset(self) -> None:
        """Clear all modal displacement and velocity states."""
        self._q.fill(0.0)
        self._v.fill(0.0)

    def process(self, force: FloatAudio) -> FloatAudio:
        """Process a finite mono force block and retain state for the next block."""
        if force.ndim != 1:
            raise ValueError("force must be one-dimensional")
        if not np.all(np.isfinite(force)):
            raise ValueError("force must contain only finite samples")
        output = np.zeros(force.size, dtype=np.float64)
        if not self.modes or force.size == 0:
            return output.astype(np.float32)
        for sample_index, sample in enumerate(force):
            excitation = float(sample)
            next_q = self._a00 * self._q + self._a01 * self._v + self._bq * excitation
            next_v = self._a10 * self._q + self._a11 * self._v + self._bv * excitation
            self._q = next_q
            self._v = next_v
            output[sample_index] = float(np.dot(self._output_gain, self._v))
        return np.asarray(output, dtype=np.float32)


def modal_response(
    force: FloatAudio,
    modes: ModeSet,
    sample_rate: int,
    *,
    tail_s: float = 0.0,
) -> FloatAudio:
    """Render one force signal, optionally appending a zero-input decay tail."""
    if not math.isfinite(tail_s) or tail_s < 0.0:
        raise ValueError("tail_s must be finite and non-negative")
    bank = ModalResonatorBank(modes, sample_rate)
    if tail_s == 0.0:
        return bank.process(force)
    tail_size = round(tail_s * sample_rate)
    extended = np.pad(force, (0, tail_size)).astype(np.float32, copy=False)
    return bank.process(extended)

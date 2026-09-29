"""
Clock models.

* :class:`TrueTime` -- the simulator's own time (the "physics" reference).
* :class:`LocalClock` -- a free-running oscillator of a device (control
  system, FPGA gateway) with a static offset, a frequency error in ppm, a
  random-walk wander, a finite timestamp resolution and read jitter.  It can
  be disciplined (stepped / frequency-steered) by a time-sync servo exactly
  like a PTP hardware clock (PHC).

``LocalClock.now_ps()`` is what firmware on that device would read.
``LocalClock.to_true(local)`` is only for test oracles (a real device cannot
know true time).
"""

from __future__ import annotations

from typing import Optional, Protocol

import numpy as np


class TimeSource(Protocol):
    def now_ps(self) -> int: ...


class TrueTime:
    """Adapter exposing a scheduler's time as a clock."""

    def __init__(self, scheduler: TimeSource):
        self._s = scheduler

    def now_ps(self) -> int:
        return self._s.now_ps()


class LocalClock:
    """Oscillator-driven device clock with PHC-style discipline hooks.

    local(t) = anchor_local + (t - anchor_true) * (1 + rate_ppm * 1e-6)

    Parameters
    ----------
    source : object with ``now_ps()`` (true time)
    offset_ps : initial local - true offset
    freq_ppm : oscillator frequency error (typ. crystal +-20 ppm, OCXO +-0.01)
    resolution_ps : timestamp quantisation (e.g. one 300 MHz cycle = 3333 ps,
        or sub-cycle with a DDMTD/IOSERDES timestamper)
    jitter_ps : rms Gaussian noise on each timestamp read (timestamp unit noise)
    wander_ppm_per_sqrt_s : random-walk frequency wander of the oscillator
    """

    def __init__(self, source: TimeSource, offset_ps: int = 0, freq_ppm: float = 0.0,
                 resolution_ps: int = 1, jitter_ps: float = 0.0,
                 wander_ppm_per_sqrt_s: float = 0.0,
                 rng: Optional[np.random.Generator] = None, name: str = "clock"):
        if resolution_ps < 1:
            raise ValueError("resolution must be >= 1 ps")
        self._src = source
        self.name = name
        self.resolution_ps = int(resolution_ps)
        self.jitter_ps = float(jitter_ps)
        self.wander = float(wander_ppm_per_sqrt_s)
        self._rng = rng if rng is not None else np.random.default_rng(0)
        t = source.now_ps()
        self._anchor_true = t
        self._anchor_local = float(t + offset_ps)
        self._osc_ppm = float(freq_ppm)      # physical oscillator error (unknown to device)
        self._steer_ppm = 0.0                # correction applied by the servo
        self._last_wander_t = t
        self.steps = 0

    # -- internal ---------------------------------------------------------
    def _rate(self) -> float:
        return 1.0 + (self._osc_ppm + self._steer_ppm) * 1e-6

    def _advance_wander(self, t: int) -> None:
        if self.wander <= 0.0 or t <= self._last_wander_t:
            return
        dt_s = (t - self._last_wander_t) * 1e-12
        # re-anchor before the rate changes so the clock stays continuous
        self._anchor_local = self._local_at(t)
        self._anchor_true = t
        self._osc_ppm += float(self._rng.normal(0.0, self.wander * np.sqrt(dt_s)))
        self._last_wander_t = t

    def _local_at(self, t: int) -> float:
        return self._anchor_local + (t - self._anchor_true) * self._rate()

    def _reanchor(self) -> None:
        t = self._src.now_ps()
        self._advance_wander(t)
        self._anchor_local = self._local_at(t)
        self._anchor_true = t

    # -- device-visible API --------------------------------------------------
    def now_ps(self) -> int:
        """Exact (noise-free, quantised) local time."""
        t = self._src.now_ps()
        self._advance_wander(t)
        v = int(self._local_at(t))
        return v - (v % self.resolution_ps)

    def timestamp(self) -> int:
        """A hardware timestamp: local time with resolution and read jitter."""
        t = self._src.now_ps()
        self._advance_wander(t)
        v = self._local_at(t)
        if self.jitter_ps > 0.0:
            v += float(self._rng.normal(0.0, self.jitter_ps))
        v = int(round(v))
        return v - (v % self.resolution_ps)

    def step(self, delta_ps: int) -> None:
        """Phase step (servo 'step' mode)."""
        self._reanchor()
        self._anchor_local += delta_ps
        self.steps += 1

    def adjust_frequency(self, steer_ppm: float) -> None:
        """Set the frequency steering applied on top of the oscillator error."""
        self._reanchor()
        self._steer_ppm = float(steer_ppm)

    @property
    def steer_ppm(self) -> float:
        return self._steer_ppm

    # -- oracle helpers (tests / telemetry only) -------------------------------
    def offset_from_true_ps(self) -> float:
        t = self._src.now_ps()
        self._advance_wander(t)
        return self._local_at(t) - t

    def effective_ppm(self) -> float:
        return self._osc_ppm + self._steer_ppm

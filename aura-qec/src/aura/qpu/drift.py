"""
Temporal drift of device parameters.

Physical effects modelled (all reproducible from the device seed):

* Ornstein-Uhlenbeck fluctuations of log(T1), qubit frequency, pi-pulse
  amplitude, CZ flux amplitude and readout IQ rotation (slow drifts seen in
  every superconducting lab; correlation times of minutes-hours).
* Two-level-system (TLS) defects: a qubit's T1 suddenly drops by a factor
  for a random duration (Poisson arrivals).
* Cosmic-ray / high-energy impacts: rare, chip-wide correlated bursts in
  which every qubit within a few coupler hops of the impact suffers a strong
  T1 suppression that recovers exponentially over ~25 ms
  (McEwen et al., Nature Physics 18, 107 (2022)).

``DriftModel.state(t)`` returns multiplicative / additive offsets at virtual
time ``t``.  Processes advance on a fixed grid (``grid_s``); the ``speedup``
factor compresses wall-lab time so drift is visible in short demos.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..core.units import S
from .device import DeviceTruth


@dataclass
class DriftState:
    t1_factor: np.ndarray        # per qubit multiplicative factor on T1
    freq_offset_hz: np.ndarray   # per qubit additive f01 offset
    amp_factor: np.ndarray       # per qubit multiplicative factor on pi_amp
    cz_amp_factor: np.ndarray    # per coupler multiplicative factor on cz_amp
    readout_rot_rad: np.ndarray  # per qubit IQ-plane rotation of the readout blobs


class _OU:
    """Vector OU process sampled on a fixed grid (exact discretisation)."""

    def __init__(self, size: int, sigma: float, tau_s: float, grid_s: float, rng: np.random.Generator):
        self.sigma = sigma
        self.a = math.exp(-grid_s / tau_s) if tau_s > 0 else 0.0
        self.b = sigma * math.sqrt(max(0.0, 1.0 - self.a ** 2))
        self.rng = rng
        self.x = rng.normal(0.0, sigma, size) if sigma > 0 else np.zeros(size)

    def step(self) -> None:
        if self.sigma > 0:
            self.x = self.a * self.x + self.b * self.rng.normal(0.0, 1.0, self.x.shape)


class DriftModel:
    def __init__(self, device: DeviceTruth, rng: np.random.Generator):
        self.dev = device
        p = device.drift
        self.p = p
        self.grid_ps = max(1, int(round(p.grid_s / p.speedup * S)))  # virtual time per grid step
        nq, nc = device.n, len(device.couplers)
        g = p.grid_s
        self._t1 = _OU(nq, p.t1_log_sigma, p.t1_tau_s, g, rng)
        self._fr = _OU(nq, p.freq_sigma_hz, p.freq_tau_s, g, rng)
        self._amp = _OU(nq, p.amp_sigma, p.amp_tau_s, g, rng)
        self._cz = _OU(nc, p.cz_amp_sigma, p.cz_amp_tau_s, g, rng)
        self._ro = _OU(nq, p.readout_rot_sigma_rad, p.readout_tau_s, g, rng)
        self.rng = rng
        self._step_index = 0
        self._tls_until = np.full(nq, -1.0)           # lab-seconds until which TLS dip is active
        self.cosmic_events: deque[tuple[float, int, np.ndarray]] = deque()   # (t_lab_s, center, affected mask)
        self.cosmic_history: list[tuple[float, int]] = []
        self._hops = self._hop_matrix()
        self._cache_index = -1
        self._cache: Optional[DriftState] = None

    def _hop_matrix(self) -> np.ndarray:
        nq = self.dev.n
        nb = self.dev.spec.neighbors()
        dist = np.full((nq, nq), 10 ** 6, dtype=np.int64)
        for s in range(nq):
            dist[s, s] = 0
            frontier = [s]
            d = 0
            while frontier and d < self.p.cosmic_radius:
                d += 1
                nxt = []
                for u in frontier:
                    for v in nb[u]:
                        if dist[s, v] > d:
                            dist[s, v] = d
                            nxt.append(v)
                frontier = nxt
        return dist

    def _lab_seconds(self, step_index: int) -> float:
        return step_index * self.p.grid_s

    def _advance_to(self, idx: int) -> None:
        p = self.p
        nq = self.dev.n
        while self._step_index < idx:
            self._step_index += 1
            for proc in (self._t1, self._fr, self._amp, self._cz, self._ro):
                proc.step()
            now = self._lab_seconds(self._step_index)
            if p.tls_rate_per_qubit_per_s > 0:
                hits = self.rng.random(nq) < p.tls_rate_per_qubit_per_s * p.grid_s
                dur = self.rng.exponential(p.tls_duration_s, nq)
                self._tls_until = np.where(hits, np.maximum(self._tls_until, now + dur), self._tls_until)
            if p.cosmic_rate_per_s > 0 and self.rng.random() < p.cosmic_rate_per_s * p.grid_s:
                center = int(self.rng.integers(nq))
                mask = self._hops[center] <= p.cosmic_radius
                self.cosmic_events.append((now, center, mask))
                self.cosmic_history.append((now, center))
            while self.cosmic_events and now - self.cosmic_events[0][0] > 10 * p.cosmic_tau_s:
                self.cosmic_events.popleft()

    def state(self, t_ps: int) -> DriftState:
        idx = int(t_ps // self.grid_ps)
        if idx == self._cache_index and self._cache is not None:
            return self._cache
        if idx < self._step_index:
            # Time only moves forward in the simulator; stale queries reuse the latest state.
            idx = self._step_index
        self._advance_to(idx)
        p = self.p
        now = self._lab_seconds(idx)
        t1f = np.exp(self._t1.x)
        t1f = np.where(self._tls_until > now, t1f * p.tls_t1_factor, t1f)
        for (t0, _c, mask) in self.cosmic_events:
            dt = now - t0
            if dt >= 0:
                supp = 1.0 + p.cosmic_t1_suppression * math.exp(-dt / p.cosmic_tau_s)
                t1f = np.where(mask, t1f / supp, t1f)
        self._cache = DriftState(t1_factor=t1f, freq_offset_hz=self._fr.x.copy(),
                                 amp_factor=1.0 + self._amp.x, cz_amp_factor=1.0 + self._cz.x,
                                 readout_rot_rad=self._ro.x.copy())
        self._cache_index = idx
        return self._cache

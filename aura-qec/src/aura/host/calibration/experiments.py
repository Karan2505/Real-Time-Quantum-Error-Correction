"""
Calibration experiments, executed on the virtual QPU's open-system physics
engine through the control system, fitted, and written to the store.

All experiments on different qubits run in parallel (as on real hardware);
coupler experiments run in parallel groups (a proper edge colouring of the
coupler graph).  Virtual time advances by the duration the experiment would
take in the lab, so the device keeps drifting while it is being calibrated.

Experiment          measures / sets
------------------  ---------------------------------------------------------
readout             IQ blobs of |0>,|1>,|2> -> discriminator, p01, p10
rabi                amplitude sweep -> pi amplitude (coarse)
rabi_fine           X90 + 2k X180 error amplification -> pi amplitude (fine)
ramsey              +- artificial detuning Ramsey -> drive frequency, T2*
t1                  inversion recovery -> T1
t2echo              Hahn echo -> T2 echo
sq_bench            X180-train decay -> residual single-qubit error
cz_phase            conditional Ramsey (both orientations) -> CZ flux amplitude,
                    dynamic-phase compensation of both qubits
cz_bench            CZ-train decay on |++> -> residual CZ error
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from ...control.readout import Discriminator, fit_discriminator
from ...core.errors import CalibrationError
from ...core.units import US
from ...qpu import pauli as Pm
from ...qpu.physics import CZPulse, Drive, Repeat, Twirled, VirtualZ, Wait
from .fitting import fit_cosine, fit_damped_cosine, fit_exp_decay, fit_line, fit_phase
from .store import CalibrationStore


@dataclass
class CalSettings:
    shots: int = 400
    readout_shots: int = 2000
    rep_delay_ps: int = 10 * US
    ramsey_max_ps: int = 8 * US
    ramsey_points: int = 81
    ramsey_art_hz: float = 1.0e6
    rabi_points: int = 41
    fine_ks: tuple[int, ...] = (0, 1, 2, 3, 4, 6, 8, 10)
    t1_points: int = 31
    echo_points: int = 31
    bench_lengths: tuple[int, ...] = (0, 100, 200, 400, 800, 1600)     # X180 pairs
    cz_bench_lengths: tuple[int, ...] = (0, 10, 20, 40, 80, 160)       # CZ pairs
    bench_shots: int = 4000
    phase_points: int = 8
    phase_shots: int = 2000


class Calibrator:
    def __init__(self, sim, qpu, store: CalibrationStore, rng: np.random.Generator,
                 settings: Optional[CalSettings] = None, log: Optional[Callable[[str], None]] = None):
        self.sim = sim
        self.qpu = qpu
        self.store = store
        self.spec = qpu.spec
        self.rng = rng
        self.cfg = settings or CalSettings()
        self.log = log or (lambda s: None)
        self.du = self.spec.durations
        self.results: dict[str, dict] = {}
        self._coupler_groups = self._color_couplers()

    # ------------------------------------------------------------------ plumbing
    def _elapse(self, points: int, shots: int, seq_ps: int, groups: int = 1) -> None:
        dt = groups * points * shots * (seq_ps + self.du.meas_ps + self.cfg.rep_delay_ps)
        self.sim.run(until_ps=self.sim.now + int(dt))

    def _disc(self, q: int) -> Discriminator:
        qc = self.store.qubits[q]
        if not self.store.readout_calibrated(q):
            raise CalibrationError(f"qubit {q}: readout not calibrated")
        return Discriminator(complex(*qc.c0), complex(*qc.c1), complex(*qc.c2) if qc.c2 else None, qc.ro_sigma)

    def _p1(self, q: int, seq, shots: Optional[int] = None, freq: Optional[float] = None) -> float:
        n = shots or self.cfg.shots
        iq = self.qpu.physics_1q(q, seq, n, drive_freq_hz=freq if freq is not None
                                 else self.store.qubits[q].drive_freq_hz)
        bits, _, _ = self._disc(q).classify(iq)
        return float(np.mean(bits))

    def _x180(self, q: int, phase: float = 0.0) -> Drive:
        return Drive(self.store.qubits[q].pi_amp, phase, self.du.x180_ps)

    def _x90(self, q: int, phase: float = 0.0) -> Drive:
        return Drive(self.store.qubits[q].sx_amp, phase, self.du.sx_ps)

    def _qubits(self, qubits) -> list[int]:
        return list(range(self.spec.num_qubits)) if qubits is None else list(qubits)

    def _groups_for(self, couplers: Optional[list[int]]) -> int:
        """Number of sequential rounds needed to run these coupler experiments in parallel
        (couplers sharing a qubit cannot run simultaneously)."""
        if couplers is None:
            return max(1, len(self._coupler_groups))
        return max(1, len(self._color_couplers(couplers)))

    def _color_couplers(self, subset: Optional[list[int]] = None) -> list[list[int]]:
        groups: list[list[int]] = []
        used: list[set] = []
        sel = self.spec.couplers if subset is None else [self.spec.couplers[i] for i in subset]
        for c in sel:
            for g, u in zip(groups, used):
                if c.q0 not in u and c.q1 not in u:
                    g.append(c.index); u.update((c.q0, c.q1))
                    break
            else:
                groups.append([c.index]); used.append({c.q0, c.q1})
        return groups

    def _freqs(self, c: int) -> tuple[float, float]:
        ci = self.spec.couplers[c]
        return self.store.qubits[ci.q0].drive_freq_hz, self.store.qubits[ci.q1].drive_freq_hz

    def _spam(self, q: int) -> tuple[float, float]:
        qc = self.store.qubits[q]
        return (qc.p01 or 0.0), (qc.p10 or 0.0)

    # ------------------------------------------------------------------ readout
    def readout(self, qubits=None) -> dict:
        n = self.cfg.readout_shots
        out = {}
        for q in self._qubits(qubits):
            qc = self.store.qubits[q]
            f = qc.drive_freq_hz
            ef_amp = qc.pi_amp / math.sqrt(2)
            iq0 = self.qpu.physics_1q(q, [], n, drive_freq_hz=f)
            iq1 = self.qpu.physics_1q(q, [self._x180(q)], n, drive_freq_hz=f)
            iq2 = self.qpu.physics_1q(q, [self._x180(q), Drive(ef_amp, 0.0, self.du.x180_ps, "ef")], n,
                                      drive_freq_hz=f)
            d = fit_discriminator(iq0, iq1, iq2)
            # assignment errors on independent shots
            b0, _, _ = d.classify(self.qpu.physics_1q(q, [], n, drive_freq_hz=f))
            b1, _, _ = d.classify(self.qpu.physics_1q(q, [self._x180(q)], n, drive_freq_hz=f))
            p01, p10 = float(np.mean(b0)), float(1 - np.mean(b1))
            self.store.update_qubit(q, self.sim.now, "readout", c0=(d.c0.real, d.c0.imag),
                                    c1=(d.c1.real, d.c1.imag), c2=(d.c2.real, d.c2.imag) if d.c2 else None,
                                    ro_sigma=d.sigma, p01=p01, p10=p10)
            out[q] = {"p01": p01, "p10": p10, "sigma": d.sigma}
        self._elapse(5, n, self.du.x180_ps * 2)
        self.results["readout"] = out
        return out

    # ------------------------------------------------------------------ rabi
    def rabi(self, qubits=None) -> dict:
        out = {}
        pts = self.cfg.rabi_points
        for q in self._qubits(qubits):
            guess = self.store.qubits[q].pi_amp
            amps = np.linspace(0, 2.0 * guess, pts)
            y = np.array([self._p1(q, [Drive(a, 0.0, self.du.x180_ps)]) for a in amps])
            fr = fit_cosine(amps, y, period_hint=2 * guess)
            pi_amp = fr.params["period"] / 2.0
            if not (0.5 * guess < pi_amp < 1.5 * guess):
                raise CalibrationError(f"qubit {q}: Rabi fit out of range ({pi_amp:.4f} vs guess {guess:.4f})")
            self.store.update_qubit(q, self.sim.now, "rabi", pi_amp=pi_amp, sx_amp=pi_amp / 2)
            out[q] = {"pi_amp": pi_amp, "rms": fr.residual_rms}
        self._elapse(pts, self.cfg.shots, self.du.x180_ps)
        self.results["rabi"] = out
        return out

    def rabi_fine(self, qubits=None, iterations: int = 2, check_only: bool = False) -> dict:
        out = {}
        ks = np.array(self.cfg.fine_ks)
        g = np.pi / 2 + 2 * np.pi * ks
        for q in self._qubits(qubits):
            eps_total = 0.0
            for _ in range(1 if check_only else iterations):
                ys = []
                for k in ks:
                    seq = [self._x90(q)] + [self._x180(q)] * (2 * int(k))
                    ys.append(self._p1(q, seq))
                p01, p10 = self._spam(q)
                contrast = max(0.2, 1 - p01 - p10)
                y = (np.array(ys) - p01) / contrast
                fl = fit_line(g, y)
                eps = 2.0 * fl.params["slope"]
                eps_total += eps
                if check_only:
                    break
                qc = self.store.qubits[q]
                new = qc.pi_amp / (1 + eps)
                self.store.update_qubit(q, self.sim.now, "rabi_fine", pi_amp=new, sx_amp=new / 2)
            out[q] = {"eps": eps_total, "pi_amp": self.store.qubits[q].pi_amp}
        self._elapse(len(ks) * (1 if check_only else iterations), self.cfg.shots,
                     self.du.x180_ps * (2 * int(ks.max()) + 1))
        self.results["rabi_fine"] = out
        return out

    # ------------------------------------------------------------------ ramsey
    def _ramsey_pair(self, q: int, art_hz: float, max_ps: int, points: int) -> tuple[float, float]:
        """Ramsey with +-artificial detuning; returns (detuning_hz, T2*_ps).
        With phase(2nd pulse) = 2 pi f_art tau the fringe frequency is |Delta + f_art|,
        so Delta = (f_plus - f_minus) / 2 provided |Delta| < f_art."""
        taus = np.linspace(0, max_ps, points).astype(np.int64)
        taus_us = taus / 1e6
        fits = []
        for sgn in (+1, -1):
            fa = sgn * art_hz
            y = np.array([self._p1(q, [self._x90(q), Wait(int(t)),
                                       self._x90(q, 2 * np.pi * fa * t * 1e-12)]) for t in taus])
            fits.append(fit_damped_cosine(taus_us, y, f_hint=art_hz / 1e6))
        delta = (fits[0].params["f"] - fits[1].params["f"]) * 1e6 / 2.0
        t2 = float(np.mean([f.params["tau"] for f in fits])) * 1e6
        return delta, t2

    def ramsey(self, qubits=None, check_only: bool = False) -> dict:
        """Two stages: a coarse wide-capture Ramsey (+-6 MHz artificial detuning,
        1.5 us) then a fine long-baseline Ramsey for kHz-level resolution and T2*."""
        c = self.cfg
        out = {}
        for q in self._qubits(qubits):
            f_drive = self.store.qubits[q].drive_freq_hz
            total = 0.0
            if not check_only:
                d1, _ = self._ramsey_pair(q, 6e6, int(1.5 * US), 61)
                total += d1
                self.store.update_qubit(q, self.sim.now, "ramsey", drive_freq_hz=f_drive + total)
            d2, t2 = self._ramsey_pair(q, c.ramsey_art_hz, c.ramsey_max_ps, c.ramsey_points)
            if check_only:
                out[q] = {"detuning_hz": d2, "t2_star_us": t2 / 1e6}
                continue
            total += d2
            self.store.update_qubit(q, self.sim.now, "ramsey", drive_freq_hz=f_drive + total, t2_star_ps=t2)
            out[q] = {"detuning_hz": total, "t2_star_us": t2 / 1e6}
        pts = 2 * c.ramsey_points + (0 if check_only else 122)
        self._elapse(pts, c.shots, int(c.ramsey_max_ps // 2))
        self.results["ramsey"] = out
        return out

    # ------------------------------------------------------------------ T1 / T2 echo
    def t1(self, qubits=None) -> dict:
        out = {}
        pts = self.cfg.t1_points
        longest = 0
        for q in self._qubits(qubits):
            guess = self.store.qubits[q].t1_ps or 60 * US
            taus = np.linspace(0, 4 * guess, pts)
            longest = max(longest, int(taus[-1]))
            y = np.array([self._p1(q, [self._x180(q), Wait(int(t))]) for t in taus])
            fr = fit_exp_decay(taus / 1e6, y)
            t1 = fr.params["tau"] * 1e6
            self.store.update_qubit(q, self.sim.now, "t1", t1_ps=t1)
            out[q] = {"t1_us": t1 / 1e6}
        self._elapse(pts, self.cfg.shots, longest // 2)
        self.results["t1"] = out
        return out

    def t2echo(self, qubits=None) -> dict:
        out = {}
        pts = self.cfg.echo_points
        longest = 0
        for q in self._qubits(qubits):
            guess = self.store.qubits[q].t1_ps or 60 * US
            taus = np.linspace(0, 3 * guess, pts)
            longest = max(longest, int(taus[-1]))
            y = []
            for t in taus:
                h = int(t // 2)
                y.append(self._p1(q, [self._x90(q), Wait(h), self._x180(q, np.pi / 2), Wait(h),
                                      self._x90(q)]))
            fr = fit_exp_decay(taus / 1e6, np.array(y))
            t2e = min(fr.params["tau"] * 1e6, 2 * (self.store.qubits[q].t1_ps or 1e30))
            self.store.update_qubit(q, self.sim.now, "t2echo", t2_echo_ps=t2e)
            out[q] = {"t2_echo_us": t2e / 1e6}
        self._elapse(pts, self.cfg.shots, longest // 2)
        self.results["t2echo"] = out
        return out

    # ------------------------------------------------------------------ benchmarks
    def sq_bench(self, qubits=None) -> dict:
        out = {}
        lens = np.array(self.cfg.bench_lengths)
        for q in self._qubits(qubits):
            x_ideal = ((0j, 1 + 0j), (1 + 0j, 0j))
            y = np.array([1 - self._p1(q, [Twirled((self._x180(q),), x_ideal, 2 * int(m))],
                                       shots=self.cfg.bench_shots) for m in lens])
            n = 2 * lens
            fr = fit_exp_decay(n.astype(float), y)
            lam = math.exp(-1.0 / fr.params["tau"]) if fr.params["tau"] > 0 else 0.0
            t1 = self.store.qubits[q].t1_ps or 60 * US
            p = 0.75 * (1 - lam * math.exp(self.du.x180_ps / t1))
            p = float(min(max(p, 0.0), 0.1))
            self.store.update_qubit(q, self.sim.now, "sq_bench", sq_error=p)
            out[q] = {"sq_error": p}
        self._elapse(len(lens), self.cfg.bench_shots, self.du.x180_ps * 2 * int(lens.max()))
        self.results["sq_bench"] = out
        return out

    # ------------------------------------------------------------------ CZ
    def _cond_phase(self, c: int, target: int, ctrl_excited: bool, phis: np.ndarray, n_cz: int = 1,
                    compensate: bool = False) -> float:
        """Ramsey phase of ``target`` after ``n_cz`` CZ pulses (optionally each followed by the
        stored virtual-Z compensation), with the other qubit in |0> or |1>."""
        ci = self.spec.couplers[c]
        tq = ci.q1 if target == 1 else ci.q0
        cq = ci.q0 if target == 1 else ci.q1
        ctrl_idx = 1 - target
        cc = self.store.couplers[c]
        disc = self._disc(tq)
        body: tuple = (CZPulse(cc.cz_amp, self.du.cz_ps),)
        if compensate:
            body = body + (VirtualZ(cc.corr0, 0), VirtualZ(cc.corr1, 1))
        ys = []
        for phi in phis:
            seq = []
            if ctrl_excited:
                seq.append(Drive(self.store.qubits[cq].pi_amp, 0.0, self.du.x180_ps, target=ctrl_idx))
            seq += [Drive(self.store.qubits[tq].sx_amp, 0.0, self.du.sx_ps, target=target),
                    Repeat(body, n_cz),
                    Drive(self.store.qubits[tq].sx_amp, float(phi), self.du.sx_ps, target=target)]
            iq = self.qpu.physics_2q(c, seq, self.cfg.phase_shots, drive_freq_hz=self._freqs(c))
            bits, _, _ = disc.classify(iq[:, target])
            ys.append(np.mean(bits))
        return fit_phase(phis, np.array(ys)).params["phi0"]

    def cz_phase(self, couplers=None, amplify: tuple[int, ...] = (5, 11)) -> dict:
        """Conditional-phase (flux amplitude) and dynamic-phase calibration with
        error amplification: odd trains of n CZs turn a phase error d into n*d."""
        cs = list(range(len(self.spec.couplers))) if couplers is None else list(couplers)
        phis = np.linspace(0, 2 * np.pi, self.cfg.phase_points, endpoint=False)
        out = {}
        runs = 0
        for c in cs:
            # coarse: single CZ, conditional phase ~ pi
            for _ in range(2):
                a0 = self._cond_phase(c, 1, False, phis)
                a1 = self._cond_phase(c, 1, True, phis)
                runs += 2
                phic = (a1 - a0) % (2 * np.pi)
                if phic < 0.3:
                    raise CalibrationError(f"coupler {c}: conditional phase {phic:.3f} rad implausible")
                cc = self.store.couplers[c]
                self.store.update_coupler(c, self.sim.now, "cz_phase", cz_amp=cc.cz_amp * math.pi / phic,
                                          cond_phase_rad=phic)
            # fine: n CZs -> n*pi + n*delta
            for n in amplify:
                a0 = self._cond_phase(c, 1, False, phis, n)
                a1 = self._cond_phase(c, 1, True, phis, n)
                runs += 2
                delta = math.remainder((a1 - a0) - n * math.pi, 2 * math.pi) / n
                cc = self.store.couplers[c]
                self.store.update_coupler(c, self.sim.now, "cz_phase", cz_amp=cc.cz_amp * math.pi / (math.pi + delta),
                                          cond_phase_rad=math.pi + delta)
            # dynamic single-qubit phases: coarse, then amplified residual with compensation applied
            f1 = self._cond_phase(c, 1, False, phis)
            f0 = self._cond_phase(c, 0, False, phis)
            runs += 2
            self.store.update_coupler(c, self.sim.now, "cz_dyn_phase", corr0=float(-f0), corr1=float(-f1))
            n = max(amplify) if amplify else 1
            for tgt in (1, 0):
                r = self._cond_phase(c, tgt, False, phis, n, compensate=True)
                runs += 1
                eps = math.remainder(r, 2 * math.pi) / n
                cc = self.store.couplers[c]
                if tgt == 1:
                    self.store.update_coupler(c, self.sim.now, "cz_dyn_phase", corr1=float(cc.corr1 - eps))
                else:
                    self.store.update_coupler(c, self.sim.now, "cz_dyn_phase", corr0=float(cc.corr0 - eps))
            cc = self.store.couplers[c]
            out[c] = {"cz_amp": cc.cz_amp, "cond_phase": cc.cond_phase_rad, "corr0": cc.corr0, "corr1": cc.corr1}
        groups = self._groups_for(couplers)
        per_coupler_runs = runs / max(1, len(cs))
        self._elapse(int(per_coupler_runs * len(phis)), self.cfg.phase_shots,
                     self.du.cz_ps * (max(amplify) if amplify else 1) + 3 * self.du.sx_ps, groups=groups)
        self.results["cz_phase"] = out
        return out

    def cz_bench(self, couplers=None) -> dict:
        cs = list(range(len(self.spec.couplers))) if couplers is None else list(couplers)
        lens = np.array(self.cfg.cz_bench_lengths)
        out = {}
        for c in cs:
            ci = self.spec.couplers[c]
            cc = self.store.couplers[c]
            d0, d1 = self._disc(ci.q0), self._disc(ci.q1)
            y = []
            for m in lens:
                seq = [Drive(self.store.qubits[ci.q0].sx_amp, 0.0, self.du.sx_ps, target=0),
                       Drive(self.store.qubits[ci.q1].sx_amp, 0.0, self.du.sx_ps, target=1)]
                cz_ideal = tuple(tuple(complex(v) for v in row) for row in np.diag([1, 1, 1, -1]))
                seq.append(Twirled((CZPulse(cc.cz_amp, self.du.cz_ps), VirtualZ(cc.corr0, 0),
                                    VirtualZ(cc.corr1, 1)), cz_ideal, 2 * int(m)))
                seq += [Drive(self.store.qubits[ci.q0].sx_amp, np.pi, self.du.sx_ps, target=0),
                        Drive(self.store.qubits[ci.q1].sx_amp, np.pi, self.du.sx_ps, target=1)]
                iq = self.qpu.physics_2q(c, seq, self.cfg.bench_shots, drive_freq_hz=self._freqs(c))
                b0, _, _ = d0.classify(iq[:, 0])
                b1, _, _ = d1.classify(iq[:, 1])
                y.append(float(np.mean(~b0 & ~b1)))
            n = 2 * lens
            fr = fit_exp_decay(n.astype(float), np.array(y))
            lam = math.exp(-1.0 / fr.params["tau"]) if fr.params["tau"] > 0 else 0.0
            p_total = 15.0 / 16.0 * (1 - lam)
            # remove the decoherence contribution during the CZ pulse (reported separately to the DEM)
            dec = 0.0
            for q in (ci.q0, ci.q1):
                qc = self.store.qubits[q]
                if qc.t1_ps and qc.t2_star_ps:
                    dec += float(Pm.pta_idle(qc.t1_ps, qc.t2_star_ps, self.du.cz_ps)[1:].sum())
            p = float(min(max(p_total - dec, 0.0), 0.2))
            self.store.update_coupler(c, self.sim.now, "cz_bench", cz_error=p)
            out[c] = {"cz_error": p, "cz_total": p_total}
        groups = self._groups_for(couplers)
        self._elapse(len(lens), self.cfg.bench_shots, self.du.cz_ps * 2 * int(lens.max()), groups=groups)
        self.results["cz_bench"] = out
        return out

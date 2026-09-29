"""
The virtual superconducting QPU.

Execution engine
----------------
* A noiseless :class:`Tableau` run of the loaded program gives the reference
  measurement record.
* Each shot (or a batch of B shots in parallel) is a :class:`FrameSimulator`
  tracking the Pauli difference from the reference, plus per-shot leakage
  flags.  Measurement truth = reference XOR frame.
* Noise is computed from the *true, drifting* device parameters combined with
  the control settings actually uploaded by the control system (drive
  frequency, pulse amplitudes, CZ flux amplitude and phase compensation), so
  mis-calibration and drift degrade QEC exactly as in a lab.
* Measurements go through the dispersive-readout physics model and come back
  as IQ points; state discrimination is the control electronics' job.

Two execution paths share the same op-level implementation:
* streaming (``begin_shot`` / ``execute``): driven by sequencer events in
  virtual time, per-shot timing (feedback delays -> extra idle noise);
* batched (``run_batch``): nominal timing, B shots vectorised, for fast
  statistics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np

from ..core.errors import HardwareFault, TimingViolation
from ..core.rng import RngFactory
from . import noise as N
from .device import DeviceSpec, DeviceTruth
from .drift import DriftModel, DriftState
from .physics import QubitPhysics, evolve_1q, evolve_2q
from .program import QpuProgram, StaticOp
from .readout import sample_iq
from .reference import reference_sample, z_eigen_annotations
from .stabilizer import FrameSimulator


@dataclass
class ControlSettings:
    """What the control electronics are actually doing to the chip."""
    drive_freq_hz: np.ndarray
    sx_amp: np.ndarray
    x180_amp: np.ndarray
    cz_amp: np.ndarray
    cz_corr: np.ndarray              # (n_couplers, 2) virtual-Z compensation per qubit of the pair
    version: int = 0

    @classmethod
    def from_spec(cls, spec: DeviceSpec) -> "ControlSettings":
        return cls(drive_freq_hz=np.array(spec.design_f01_hz, float),
                   sx_amp=np.array(spec.design_pi_amp, float) / 2.0,
                   x180_amp=np.array(spec.design_pi_amp, float),
                   cz_amp=np.array(spec.design_cz_amp, float),
                   cz_corr=np.zeros((len(spec.couplers), 2)))


def _wrap(a: np.ndarray | float) -> np.ndarray | float:
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


class _Shots:
    """State of B shots executing the same program in lock-step."""

    def __init__(self, n: int, batch: int, rng: np.random.Generator, t0: int):
        self.frames = FrameSimulator(n, batch, rng)
        self.leaked = np.zeros((n, batch), dtype=bool)
        self.last_end = np.full(n, t0, dtype=np.int64)
        self.busy_until = np.full(n, t0, dtype=np.int64)
        self.next_ptr = np.zeros(n, dtype=np.int64)
        self.iq: dict[int, np.ndarray] = {}
        self.true_bits: dict[int, np.ndarray] = {}
        self.leak_at_meas: dict[int, np.ndarray] = {}
        self.batch = batch


class VirtualQPU:
    def __init__(self, device: DeviceTruth, rng_factory: RngFactory, scheduler=None,
                 readout_sink: Optional[Callable[..., None]] = None):
        self.dev = device
        self.spec = device.spec
        self.n = device.n
        self.rngf = rng_factory
        self.rng = rng_factory.stream("qpu/noise")
        self._ro_rng = rng_factory.stream("qpu/readout")
        self._phys_rng = rng_factory.stream("qpu/physics")
        self.drift = DriftModel(device, rng_factory.stream("qpu/drift"))
        self.sched = scheduler
        self.readout_sink = readout_sink
        self.settings = ControlSettings.from_spec(self.spec)
        self._neighbors = self.spec.neighbors()
        self._coupler_idx = self.spec.coupler_index()
        self._snap_key = None
        self._snap: Optional[N.NoiseSnapshot] = None
        self._chan_cache: dict = {}
        self.program: Optional[QpuProgram] = None
        self._ref_bits: Optional[np.ndarray] = None
        self._zann: list[tuple] = []
        self._per_qubit_ops: list[list[int]] = []
        self._shot: Optional[_Shots] = None
        self._shot_id: int = -1
        self.shots_executed = 0
        self.ops_executed = 0

    # ================================================================ control-facing configuration
    def upload_settings(self, settings: ControlSettings) -> None:
        for nm, arr, size in (("drive_freq_hz", settings.drive_freq_hz, self.n),
                              ("sx_amp", settings.sx_amp, self.n), ("x180_amp", settings.x180_amp, self.n),
                              ("cz_amp", settings.cz_amp, len(self.dev.couplers))):
            if np.shape(arr) != (size,):
                raise HardwareFault(f"control settings {nm} has shape {np.shape(arr)}, expected ({size},)")
        if np.shape(settings.cz_corr) != (len(self.dev.couplers), 2):
            raise HardwareFault("control settings cz_corr has wrong shape")
        self.settings = settings
        self._snap_key = None
        self._chan_cache.clear()

    def oracle_control_settings(self, t_ps: Optional[int] = None) -> ControlSettings:
        """Perfect control settings computed from the hidden truth at time t.

        TEST / BENCHMARK ONLY: a real lab can never do this; production code
        must use calibration.  Useful to separate calibration quality from
        everything else."""
        t = self._now() if t_ps is None else t_ps
        d = self.drift.state(t)
        pi = np.array([qt.pi_amp for qt in self.dev.qubits]) * d.amp_factor
        f = np.array([qt.f01_hz for qt in self.dev.qubits]) + d.freq_offset_hz
        cz = np.array([ct.cz_amp for ct in self.dev.couplers]) * d.cz_amp_factor
        corr = -np.array([[ct.dyn_phase0, ct.dyn_phase1] for ct in self.dev.couplers]).reshape(-1, 2)
        return ControlSettings(drive_freq_hz=f, sx_amp=pi / 2.0, x180_amp=pi, cz_amp=cz, cz_corr=corr,
                               version=10 ** 9 + int(t // max(1, self.drift.grid_ps)))

    # ================================================================ effective physics
    def _now(self) -> int:
        return self.sched.now_ps() if self.sched is not None else 0

    def physics_state(self, t_ps: int) -> DriftState:
        return self.drift.state(t_ps)

    def qubit_physics(self, q: int, t_ps: int, drive_freq_hz: Optional[float] = None) -> QubitPhysics:
        d = self.drift.state(t_ps)
        qt = self.dev.qubits[q]
        t1, t2s, _ = self._coherence(q, d)
        f = self.settings.drive_freq_hz[q] if drive_freq_hz is None else drive_freq_hz
        return QubitPhysics(t1_ps=t1, t2_ps=t2s, detuning_hz=qt.f01_hz + d.freq_offset_hz[q] - f,
                            anharm_hz=qt.anharm_hz, pi_amp=qt.pi_amp * d.amp_factor[q],
                            ef_pi_amp=qt.ef_pi_amp * d.amp_factor[q], sq_depol=qt.sq_depol)

    def _coherence(self, q: int, d: DriftState) -> tuple[float, float, float]:
        qt = self.dev.qubits[q]
        t1b = qt.t1_ps
        t1 = t1b * float(d.t1_factor[q])

        def rescale(t2b: float) -> float:
            inv_phi = max(0.0, 1.0 / t2b - 0.5 / t1b)
            return min(2 * t1, 1.0 / (0.5 / t1 + inv_phi))
        return t1, rescale(qt.t2_star_ps), rescale(qt.t2_echo_ps)

    def snapshot(self, t_ps: int) -> N.NoiseSnapshot:
        """True noise parameters at time t given the uploaded control settings."""
        key = (int(t_ps // self.drift.grid_ps), self.settings.version, id(self.settings))
        if key == self._snap_key and self._snap is not None:
            return self._snap
        d = self.drift.state(t_ps)
        s = self.settings
        qs = []
        for q, qt in enumerate(self.dev.qubits):
            t1, t2s, t2e = self._coherence(q, d)
            pi_amp = qt.pi_amp * float(d.amp_factor[q])
            qs.append(N.QubitNoise(
                t1_ps=t1, t2_star_ps=t2s, t2_echo_ps=t2e, sq_depol=qt.sq_depol,
                overrot_sx=float(s.sx_amp[q] / (pi_amp / 2.0) - 1.0),
                overrot_x180=float(s.x180_amp[q] / pi_amp - 1.0),
                detuning_hz=float(qt.f01_hz + d.freq_offset_hz[q] - s.drive_freq_hz[q]),
                reset_error=qt.reset_error))
        cs = {}
        for c, ci in enumerate(self.spec.couplers):
            ct = self.dev.couplers[c]
            amp_true = ct.cz_amp * float(d.cz_amp_factor[c])
            cs[(ci.q0, ci.q1)] = N.CouplerNoise(
                cz_depol=ct.cz_depol, cz_phase_err_rad=float(math.pi * (s.cz_amp[c] / amp_true - 1.0)),
                phase_err_a=float(_wrap(ct.dyn_phase0 + s.cz_corr[c, 0])),
                phase_err_b=float(_wrap(ct.dyn_phase1 + s.cz_corr[c, 1])),
                spectator_z=ct.spectator_z)
        du = self.spec.durations
        self._snap = N.build_snapshot(qs, cs, self._neighbors, du.sx_ps, du.x180_ps, du.cz_ps)
        self._snap_key = key
        self._chan_cache.clear()
        return self._snap

    # ================================================================ program loading
    def load_program(self, program: QpuProgram) -> None:
        program.validate()
        if program.num_qubits != self.n:
            raise HardwareFault(f"program is for {program.num_qubits} qubits, device has {self.n}")
        for o in program.ops:
            if o.kind == "CZ" and (o.qubits[0], o.qubits[1]) not in self._coupler_idx:
                raise HardwareFault(f"op {o.index}: no coupler between qubits {o.qubits}")
        if self.program is not None and self.program.fingerprint() == program.fingerprint():
            return
        self.program = program
        self._ref_bits = self._reference(program)
        self._zann = z_eigen_annotations(program, self._ref_bits)
        per_q: list[list[int]] = [[] for _ in range(self.n)]
        for o in program.ops:
            for q in o.qubits:
                per_q[q].append(o.index)
        self._per_qubit_ops = per_q

    def _reference(self, program: QpuProgram) -> np.ndarray:
        return reference_sample(program, self.rngf.stream("qpu/reference/" + program.fingerprint()))

    def reference_bits(self) -> np.ndarray:
        if self._ref_bits is None:
            raise HardwareFault("no program loaded")
        return self._ref_bits.copy()

    # ================================================================ op execution (shared)
    @staticmethod
    def _prep(probs: np.ndarray) -> tuple[np.ndarray, float]:
        """Channel -> (cumulative non-identity probabilities, total error probability)."""
        c = np.cumsum(np.asarray(probs, float)[1:])
        return c, float(c[-1]) if len(c) else 0.0

    def _sample_apply(self, st: _Shots, qubits: tuple[int, ...], prepped: tuple[np.ndarray, float]) -> None:
        cum, perr = prepped
        if perr <= 0.0:
            return
        u = self.rng.random(st.batch)
        hit = u < perr
        if not hit.any():
            return
        idx = np.zeros(st.batch, dtype=np.int64)
        idx[hit] = np.minimum(np.searchsorted(cum, u[hit], side="right"), len(cum) - 1) + 1
        st.frames.apply_pauli_indices(qubits, idx)

    def _cache_put(self, key, val):
        if len(self._chan_cache) > 200_000:
            self._chan_cache.clear()
        self._chan_cache[key] = val
        return val

    def _idle_channel(self, snap: N.NoiseSnapshot, q: int, gap: int, echoed: bool) -> tuple[np.ndarray, float]:
        key = (q, gap, echoed)
        ch = self._chan_cache.get(key)
        if ch is None:
            ch = self._cache_put(key, self._prep(N.idle_channel(snap.qubits[q], gap, echoed)))
        return ch

    def _decay_prob(self, snap: N.NoiseSnapshot, q: int, gap: int) -> float:
        key = ("ad", q, gap)
        g = self._chan_cache.get(key)
        if g is None:
            g = self._cache_put(key, 1.0 - math.exp(-gap / snap.qubits[q].t1_ps))
        return g

    def _gate_noise(self, snap: N.NoiseSnapshot, kind: str, qubits: tuple[int, ...]) -> list:
        key = ("g", kind, qubits)
        locs = self._chan_cache.get(key)
        if locs is None:
            locs = self._cache_put(key, [(loc.qubits, self._prep(loc.probs)) for loc in N.op_noise(kind, qubits, snap)])
        return locs

    def _apply_op(self, st: _Shots, o: StaticOp, t: int, snap: N.NoiseSnapshot) -> None:
        fr = st.frames
        b = st.batch
        # ---- timing and idle noise ------------------------------------------
        for q in o.qubits:
            if t < st.busy_until[q]:
                raise TimingViolation(f"op {o.index} ({o.kind}) starts at {t} ps but qubit {q} "
                                      f"is busy until {st.busy_until[q]} ps")
            gap = int(t - st.last_end[q])
            if gap > 0:
                # seepage out of |2> (decay to |1>, rate ~2/T1)
                lk = st.leaked[q]
                if lk.any():
                    ps = 1.0 - math.exp(-gap / (snap.qubits[q].t1_ps / 2.0))
                    back = lk & (self.rng.random(b) < ps)
                    if back.any():
                        st.leaked[q] &= ~back
                        fr.fx[q] ^= back & (self.rng.random(b) < 0.5)
                        fr.fz[q] ^= back & (self.rng.random(b) < 0.5)
                z = self._zann[o.index][o.qubits.index(q)]
                if z >= 0:
                    # computational-basis state: exact amplitude damping (|0> never excites)
                    gamma = self._decay_prob(snap, q, gap)
                    excited = (fr.fx[q] ^ bool(z)) & ~st.leaked[q]
                    if excited.any():
                        fr.fx[q] ^= excited & (self.rng.random(b) < gamma)
                else:
                    self._sample_apply(st, (q,), self._idle_channel(snap, q, gap, o.echoed))
        # ---- the operation ---------------------------------------------------
        k = o.kind
        if k == "SX":
            q = o.qubits[0]
            if o.phase_q % 2:
                fr.s([q]); fr.sx([q]); fr.s([q])
            else:
                fr.sx([q])
            if o.conditional:
                raise HardwareFault("conditional SX is not a Pauli; only conditional X180 is supported")
        elif k == "X180":
            q = o.qubits[0]
            if o.conditional:   # executed at run time but absent from the reference
                fr.fx[q] ^= True
                if o.phase_q % 2:
                    fr.fz[q] ^= True
        elif k == "CZ":
            a, c = o.qubits
            fr.cz([a], [c])
            la, lc = st.leaked[a], st.leaked[c]
            for victim, mask in ((c, la & ~lc), (a, lc & ~la)):
                if mask.any():
                    rnd = self.rng.integers(0, 4, size=b)
                    fr.fx[victim] ^= mask & ((rnd == 1) | (rnd == 2))
                    fr.fz[victim] ^= mask & ((rnd == 2) | (rnd == 3))
        elif k == "MEASURE":
            q = o.qubits[0]
            ref = bool(self._ref_bits[o.index])
            bit = fr.fx[q] ^ ref
            states = np.where(st.leaked[q], 2, bit.astype(np.int64))
            rot = float(self.drift.state(t).readout_rot_rad[q])
            iq, post = sample_iq(states, self.dev.qubits[q].readout, snap.qubits[q].t1_ps,
                                 o.duration_ps, rot, self._ro_rng)
            fr.measure_z(q)
            normal = ~st.leaked[q]
            flip = normal & (post != bit.astype(np.int64))
            fr.fx[q] ^= flip
            returned = st.leaked[q] & (post == 1)
            if returned.any():
                # leaked qubit decayed to |1> during readout: post state |1>, frame x = 1 XOR ref
                st.leaked[q] &= ~returned
                fr.fx[q] = np.where(returned, not ref, fr.fx[q])
            st.iq[o.acq] = iq
            st.true_bits[o.acq] = bit.copy()
            st.leak_at_meas[o.acq] = states == 2
        elif k == "RESET":
            q = o.qubits[0]
            fr.reset_z(q)
            lk = st.leaked[q]
            if lk.any():
                removed = lk & (self.rng.random(b) < self.dev.qubits[q].reset_leak_removal)
                st.leaked[q] &= ~removed
        else:  # pragma: no cover - validated earlier
            raise HardwareFault(f"unknown op {k}")
        # ---- gate noise -------------------------------------------------------
        if k != "MEASURE":
            for qs, prepped in self._gate_noise(snap, k, o.qubits):
                self._sample_apply(st, qs, prepped)
        # ---- leakage creation ---------------------------------------------------
        if k in ("SX", "X180"):
            p = self.dev.qubits[o.qubits[0]].leak_sq
            if p:
                st.leaked[o.qubits[0]] |= self.rng.random(b) < p
        elif k == "CZ":
            p = self.dev.couplers[self._coupler_idx[(o.qubits[0], o.qubits[1])]].leak_cz
            if p:
                for q in o.qubits:
                    st.leaked[q] |= self.rng.random(b) < p
        for q in o.qubits:
            st.last_end[q] = t + o.duration_ps
            st.busy_until[q] = t + o.duration_ps
        self.ops_executed += 1

    # ================================================================ streaming API
    def begin_shot(self, shot_id: int) -> None:
        if self.program is None:
            raise HardwareFault("no program loaded")
        if self._shot is not None:
            raise HardwareFault(f"shot {self._shot_id} still active")
        self._shot = _Shots(self.n, 1, self.rng, self._now())
        self._shot_id = shot_id

    def execute(self, ops: Sequence[StaticOp]) -> None:
        """Apply ops that start *now* (called by sequencers in virtual time)."""
        st = self._shot
        if st is None:
            raise HardwareFault("execute() outside a shot")
        t = self._now()
        snap = self.snapshot(t)
        for o in ops:
            ptrs = self._check_order(st, o)      # validated first, committed only on success
            self._apply_op(st, o, t, snap)
            for q, p in ptrs:
                st.next_ptr[q] = p
            if o.kind == "MEASURE" and self.readout_sink is not None:
                iq = complex(st.iq[o.acq][0])
                self.sched.call_later(o.duration_ps, self.readout_sink, self._shot_id, o.acq,
                                      o.qubits[0], iq)

    def _check_order(self, st: _Shots, o: StaticOp) -> list[tuple[int, int]]:
        prog = self.program
        out = []
        for q in o.qubits:
            lst = self._per_qubit_ops[q]
            p = int(st.next_ptr[q])
            while p < len(lst) and lst[p] != o.index:
                if not prog.ops[lst[p]].conditional:
                    raise HardwareFault(f"qubit {q}: expected op {lst[p]} ({prog.ops[lst[p]].kind}) "
                                        f"but received op {o.index} ({o.kind})")
                p += 1
            if p >= len(lst):
                raise HardwareFault(f"qubit {q}: unexpected op {o.index}")
            out.append((q, p + 1))
        return out

    def end_shot(self) -> dict:
        st = self._shot
        if st is None:
            raise HardwareFault("end_shot() without an active shot")
        self._shot = None
        self.shots_executed += 1
        return {"true_bits": {a: bool(v[0]) for a, v in st.true_bits.items()},
                "leaked": {a: bool(v[0]) for a, v in st.leak_at_meas.items()}}

    @property
    def shot_active(self) -> bool:
        return self._shot is not None

    # ================================================================ batched API
    def run_batch(self, shots: int, t_start_ps: Optional[int] = None, batch: int = 1024) -> dict:
        """Execute the loaded program for ``shots`` shots with nominal timing.

        Returns {'iq': (shots, num_acq) complex, 'true_bits': (shots, num_acq) bool,
        'leaked': (shots, num_acq) bool}.
        """
        prog = self.program
        if prog is None:
            raise HardwareFault("no program loaded")
        if any(o.conditional for o in prog.ops):
            raise HardwareFault("batched execution does not support feedback; use streaming mode")
        t0 = self._now() if t_start_ps is None else t_start_ps
        iq = np.zeros((shots, prog.num_acq), dtype=complex)
        tb = np.zeros((shots, prog.num_acq), dtype=bool)
        lk = np.zeros((shots, prog.num_acq), dtype=bool)
        done = 0
        while done < shots:
            b = min(batch, shots - done)
            st = _Shots(self.n, b, self.rng, t0)
            for o in prog.ops:
                t = t0 + o.t_nominal_ps
                self._apply_op(st, o, t, self.snapshot(t))
            for a in range(prog.num_acq):
                iq[done:done + b, a] = st.iq[a]
                tb[done:done + b, a] = st.true_bits[a]
                lk[done:done + b, a] = st.leak_at_meas[a]
            done += b
            self.shots_executed += b
        return {"iq": iq, "true_bits": tb, "leaked": lk}

    # ================================================================ calibration physics
    def physics_1q(self, q: int, seq, shots: int, drive_freq_hz: Optional[float] = None,
                   t_ps: Optional[int] = None) -> np.ndarray:
        """Run a single-qubit pulse sequence ending in a measurement; returns IQ points."""
        t = self._now() if t_ps is None else t_ps
        phys = self.qubit_physics(q, t, drive_freq_hz)
        pops = evolve_1q(phys, seq)
        states = self._phys_rng.choice(3, size=shots, p=pops)
        rot = float(self.drift.state(t).readout_rot_rad[q])
        iq, _ = sample_iq(states, self.dev.qubits[q].readout, phys.t1_ps, self.spec.durations.meas_ps,
                          rot, self._ro_rng)
        return iq

    def physics_2q(self, coupler: int, seq, shots: int, drive_freq_hz: Optional[tuple[float, float]] = None,
                   t_ps: Optional[int] = None) -> np.ndarray:
        """Two-qubit sequence on a coupler's qubits; returns IQ of shape (shots, 2).
        ``drive_freq_hz``: drive-frame frequencies of (q0, q1) (default: uploaded settings)."""
        t = self._now() if t_ps is None else t_ps
        ci = self.spec.couplers[coupler]
        ct = self.dev.couplers[coupler]
        d = self.drift.state(t)
        f0, f1 = drive_freq_hz if drive_freq_hz is not None else (None, None)
        p0 = self.qubit_physics(ci.q0, t, f0)
        p1 = self.qubit_physics(ci.q1, t, f1)
        amp_true = ct.cz_amp * float(d.cz_amp_factor[coupler])
        pops = evolve_2q(p0, p1, seq, amp_true, (ct.dyn_phase0, ct.dyn_phase1), ct.cz_depol)
        outcome = self._phys_rng.choice(4, size=shots, p=pops)
        s0 = outcome // 2
        s1 = outcome % 2
        out = np.zeros((shots, 2), dtype=complex)
        for j, (qq, ss, ph) in enumerate(((ci.q0, s0, p0), (ci.q1, s1, p1))):
            rot = float(d.readout_rot_rad[qq])
            out[:, j], _ = sample_iq(ss, self.dev.qubits[qq].readout, ph.t1_ps,
                                     self.spec.durations.meas_ps, rot, self._ro_rng)
        return out

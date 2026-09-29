"""
Open-system (Lindblad) simulation of one or two transmons, used for
calibration experiments that a Pauli / stabilizer model cannot represent
(Rabi oscillations, Ramsey fringes, T1/T2 decay curves, conditional phase).

One qubit is a qutrit (|0>,|1>,|2>) so readout of |2> and ef-pulses work.
Two-qubit experiments use a 4-level (two-qubit) space.

Model
-----
H (drive frame) = 2 pi [ Delta n + alpha P_2 ]          (qutrit)
L_1 = sqrt(1/T1) |0><1| ,  L_2 = sqrt(2/T1) |1><2| ,  L_phi = sqrt(2/Tphi) n
with 1/T2 = 1/(2 T1) + 1/Tphi.  Pulses are applied as ideal rotations of the
realised angle (amplitude / true pi-amplitude * pi) sandwiched between
half-duration free evolutions, so detuning and decoherence act during pulses.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence, Union

import numpy as np
from scipy.linalg import expm

# ------------------------------------------------------------------ elements
@dataclass(frozen=True)
class Drive:
    amp: float                  # control amplitude (same units as pi_amp)
    phase_rad: float = 0.0
    duration_ps: int = 24_000
    transition: str = "ge"      # 'ge' or 'ef'
    target: int = 0             # qubit index within the experiment (0 or 1)


@dataclass(frozen=True)
class Wait:
    duration_ps: int


@dataclass(frozen=True)
class VirtualZ:
    angle_rad: float
    target: int = 0


@dataclass(frozen=True)
class CZPulse:
    amp: float
    duration_ps: int = 48_000


@dataclass(frozen=True)
class Repeat:
    """``body`` repeated ``n`` times, evaluated as (superoperator)^n -- exact and
    O(log n).  The body may shift the frame (virtual Z) only if it contains no
    drive pulse; the accumulated frame shift is applied after the block."""
    body: tuple
    n: int


@dataclass(frozen=True)
class Twirled:
    """``body`` (a Clifford operation, e.g. an X180 pulse or a compensated CZ)
    Pauli-twirled and repeated ``n`` times -- the ensemble average over random
    Pauli frames (randomised compiling / randomised benchmarking style).
    ``ideal`` is the ideal unitary of the body on the qubit space (2x2 or 4x4);
    the twirling Paulis are ideal (virtual frame randomisation).  Coherent
    errors are converted into stochastic Pauli errors, so the decay measures
    the average gate error rather than a coherent build-up."""
    body: tuple
    ideal: tuple            # nested tuples of complex (hashable)
    n: int


Element = Union[Drive, Wait, VirtualZ, CZPulse, Repeat, Twirled]


def _unitary_super(u: np.ndarray) -> np.ndarray:
    return np.kron(u.conj(), u)


def _pauli_set(nq: int, dim: int) -> list[np.ndarray]:
    ps = [np.eye(2, dtype=complex)] + _PAULIS
    if nq == 1:
        out = []
        for p in ps:
            if dim == 3:
                u = np.eye(3, dtype=complex)
                u[:2, :2] = p
                out.append(u)
            else:
                out.append(p)
        return out
    return [np.kron(a, b) for a in ps for b in ps]


def _twirl_super(body_super: np.ndarray, ideal: np.ndarray, nq: int, dim: int) -> np.ndarray:
    uid = ideal
    if dim == 3 and uid.shape[0] == 2:
        u3 = np.eye(3, dtype=complex)
        u3[:2, :2] = uid
        uid = u3
    acc = np.zeros_like(body_super)
    paulis = _pauli_set(nq, dim)
    for p in paulis:
        pc = uid @ p @ uid.conj().T                 # P' = U P U^dag (Pauli, since U is Clifford)
        acc += _unitary_super(pc.conj().T) @ body_super @ _unitary_super(p)
    return acc / len(paulis)


@dataclass
class QubitPhysics:
    """Instantaneous physical parameters of one transmon (truth + drift)."""
    t1_ps: float
    t2_ps: float                # T2* (Ramsey)
    detuning_hz: float          # f01(true) - f_drive
    anharm_hz: float
    pi_amp: float
    ef_pi_amp: float
    sq_depol: float = 0.0       # residual depolarising error per drive pulse (qubit subspace)


def _lindblad(h: np.ndarray, ls: Sequence[np.ndarray]) -> np.ndarray:
    """Liouvillian (column-stacked vec convention: vec(A X B) = (B^T kron A) vec(X))."""
    d = h.shape[0]
    eye = np.eye(d)
    lv = -1j * (np.kron(eye, h) - np.kron(h.T, eye))
    for l in ls:
        ld = l.conj().T @ l
        lv += np.kron(l.conj(), l) - 0.5 * (np.kron(eye, ld) + np.kron(ld.T, eye))
    return lv


def _tphi(t1: float, t2: float) -> float:
    inv = 1.0 / t2 - 0.5 / t1
    return math.inf if inv <= 0 else 1.0 / inv


@lru_cache(maxsize=4096)
def _qutrit_propagator(t1: float, t2: float, det: float, anh: float, dur: int) -> np.ndarray:
    t1s, dur_s = t1 * 1e-12, dur * 1e-12
    tphi = _tphi(t1, t2) * 1e-12
    n = np.diag([0.0, 1.0, 2.0])
    h = 2 * math.pi * (det * n + anh * np.diag([0.0, 0.0, 1.0]))
    ls = []
    if math.isfinite(t1s):
        a01 = np.zeros((3, 3)); a01[0, 1] = 1.0
        a12 = np.zeros((3, 3)); a12[1, 2] = 1.0
        ls += [math.sqrt(1 / t1s) * a01, math.sqrt(2 / t1s) * a12]
    if math.isfinite(tphi):
        ls.append(math.sqrt(2 / tphi) * n)
    return expm(_lindblad(h, ls) * dur_s)


def _qubit_ops(t1: float, t2: float, det: float):
    t1s = t1 * 1e-12
    tphi = _tphi(t1, t2) * 1e-12
    n = np.diag([0.0, 1.0])
    h = 2 * math.pi * det * n
    ls = []
    if math.isfinite(t1s):
        a = np.array([[0.0, 1.0], [0.0, 0.0]])
        ls.append(math.sqrt(1 / t1s) * a)
    if math.isfinite(tphi):
        ls.append(math.sqrt(2 / tphi) * n)
    return h, ls


@lru_cache(maxsize=4096)
def _two_qubit_propagator(p0: tuple, p1: tuple, dur: int) -> np.ndarray:
    h0, l0 = _qubit_ops(*p0)
    h1, l1 = _qubit_ops(*p1)
    e = np.eye(2)
    h = np.kron(h0, e) + np.kron(e, h1)
    ls = [np.kron(l, e) for l in l0] + [np.kron(e, l) for l in l1]
    return expm(_lindblad(h, ls) * dur * 1e-12)


def _rot(theta: float, phi: float) -> np.ndarray:
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    return np.array([[c, -1j * s * np.exp(-1j * phi)], [-1j * s * np.exp(1j * phi), c]])


def _embed(u2: np.ndarray, levels: tuple[int, int], dim: int = 3) -> np.ndarray:
    u = np.eye(dim, dtype=complex)
    i, j = levels
    u[np.ix_([i, j], [i, j])] = u2
    return u


_PAULIS = [np.array([[0, 1], [1, 0]], complex), np.array([[0, -1j], [1j, 0]], complex),
           np.array([[1, 0], [0, -1]], complex)]


def _depolarize_qubit_subspace(rho: np.ndarray, p: float, dim: int = 3) -> np.ndarray:
    if p <= 0:
        return rho
    out = (1 - p) * rho
    for pm in _PAULIS:
        u = _embed(pm, (0, 1), dim) if dim == 3 else pm
        if dim == 3:
            u[2, 2] = 1.0
        out = out + (p / 3) * (u @ rho @ u.conj().T)
    return out


def _depolarize_2q(rho: np.ndarray, p: float) -> np.ndarray:
    if p <= 0:
        return rho
    ps = [np.eye(2, dtype=complex)] + _PAULIS
    out = (1 - p) * rho
    for i in range(4):
        for j in range(4):
            if i == 0 and j == 0:
                continue
            u = np.kron(ps[i], ps[j])
            out = out + (p / 15) * (u @ rho @ u.conj().T)
    return out


def _depolarize_1q_in_2q(rho: np.ndarray, p: float, target: int) -> np.ndarray:
    if p <= 0:
        return rho
    e = np.eye(2)
    out = (1 - p) * rho
    for pm in _PAULIS:
        u = np.kron(pm, e) if target == 0 else np.kron(e, pm)
        out = out + (p / 3) * (u @ rho @ u.conj().T)
    return out


def _apply_super(prop: np.ndarray, rho: np.ndarray) -> np.ndarray:
    d = rho.shape[0]
    return (prop @ rho.reshape(-1, order="F")).reshape(d, d, order="F")


def _step_1q(q: QubitPhysics, key: tuple, el, rho: np.ndarray, frame: float) -> tuple[np.ndarray, float]:
    if isinstance(el, Wait):
        if el.duration_ps > 0:
            rho = _apply_super(_qutrit_propagator(*key, int(el.duration_ps)), rho)
    elif isinstance(el, VirtualZ):
        frame += el.angle_rad
    elif isinstance(el, Drive):
        half = int(el.duration_ps // 2)
        if half:
            rho = _apply_super(_qutrit_propagator(*key, half), rho)
        if el.transition == "ge":
            theta = math.pi * el.amp / q.pi_amp
            u = _embed(_rot(theta, el.phase_rad - frame), (0, 1))
        else:
            theta = math.pi * el.amp / q.ef_pi_amp
            u = _embed(_rot(theta, el.phase_rad - frame), (1, 2))
        rho = u @ rho @ u.conj().T
        if el.transition == "ge":
            rho = _depolarize_qubit_subspace(rho, q.sq_depol)
        rest = int(el.duration_ps) - half
        if rest:
            rho = _apply_super(_qutrit_propagator(*key, rest), rho)
    elif isinstance(el, Repeat):
        rho, frame = _repeat(lambda r, f: _run(lambda e, rr, ff: _step_1q(q, key, e, rr, ff), el.body, r, f),
                             el, rho, frame)
    elif isinstance(el, Twirled):
        sup = _block_super(lambda r: _run(lambda e, rr, ff: _step_1q(q, key, e, rr, ff), el.body, r, frame)[0], 3)
        tw = _twirl_super(sup, np.array(el.ideal, dtype=complex), 1, 3)
        rho = (np.linalg.matrix_power(tw, el.n) @ rho.reshape(-1, order="F")).reshape(3, 3, order="F")
    else:
        raise TypeError(f"element {el!r} not valid in a single-qubit sequence")
    return rho, frame


def _block_super(apply, d: int) -> np.ndarray:
    cols = []
    for k in range(d * d):
        e = np.zeros(d * d, dtype=complex)
        e[k] = 1.0
        cols.append(apply(e.reshape(d, d, order="F")).reshape(-1, order="F"))
    return np.column_stack(cols)


def _run(step, seq, rho, frame):
    for el in seq:
        rho, frame = step(el, rho, frame)
    return rho, frame


def _repeat(block, el: "Repeat", rho: np.ndarray, frame: float) -> tuple[np.ndarray, float]:
    if el.n <= 0:
        return rho, frame
    if el.n <= 3:
        for _ in range(el.n):
            rho, frame = block(rho, frame)
        return rho, frame
    has_drive = any(isinstance(e, (Drive, Repeat)) for e in el.body)
    dz = sum(e.angle_rad for e in el.body if isinstance(e, VirtualZ))
    if has_drive and abs(math.remainder(dz, 2 * math.pi)) > 1e-12:
        for _ in range(el.n):                    # frame-dependent body: unroll
            rho, frame = block(rho, frame)
        return rho, frame
    d = rho.shape[0]
    cols = []
    for k in range(d * d):
        e = np.zeros(d * d, dtype=complex)
        e[k] = 1.0
        out, _ = block(e.reshape(d, d, order="F"), frame)
        cols.append(out.reshape(-1, order="F"))
    sup = np.column_stack(cols)
    rho = (np.linalg.matrix_power(sup, el.n) @ rho.reshape(-1, order="F")).reshape(d, d, order="F")
    return rho, frame + el.n * dz


def evolve_1q(q: QubitPhysics, seq: Sequence[Element]) -> np.ndarray:
    """Final qutrit populations [p0, p1, p2] after the sequence (starting in |0>)."""
    rho = np.zeros((3, 3), dtype=complex)
    rho[0, 0] = 1.0
    key = (float(q.t1_ps), float(q.t2_ps), float(q.detuning_hz), float(q.anharm_hz))
    rho, _ = _run(lambda e, r, f: _step_1q(q, key, e, r, f), seq, rho, 0.0)
    pops = np.clip(np.real(np.diag(rho)), 0.0, None)
    return pops / pops.sum()


def evolve_2q(q0: QubitPhysics, q1: QubitPhysics, seq: Sequence[Element], cz_true_amp: float,
              dyn_phase: tuple[float, float], cz_depol: float = 0.0) -> np.ndarray:
    """Populations of |00>,|01>,|10>,|11> (qubit 0 most significant).

    CZPulse realises diag(1, e^{i f1}, e^{i f0}, e^{i (f0+f1+phi_c)}) with
    conditional phase phi_c = pi * amp / cz_true_amp and dynamic single-qubit
    phases ``dyn_phase`` (f0, f1)."""
    rho = np.zeros((4, 4), dtype=complex)
    rho[0, 0] = 1.0
    p0 = (float(q0.t1_ps), float(q0.t2_ps), float(q0.detuning_hz))
    p1 = (float(q1.t1_ps), float(q1.t2_ps), float(q1.detuning_hz))
    e = np.eye(2)

    def free(r, dur: int):
        if dur > 0:
            r = _apply_super(_two_qubit_propagator(p0, p1, int(dur)), r)
        return r

    def step(el, r, frames):
        if isinstance(el, Wait):
            return free(r, el.duration_ps), frames
        if isinstance(el, VirtualZ):
            f = list(frames)
            f[el.target] += el.angle_rad
            return r, tuple(f)
        if isinstance(el, Drive):
            if el.transition != "ge":
                raise ValueError("two-qubit sequences support ge drives only")
            half = int(el.duration_ps // 2)
            r = free(r, half)
            qp = q0 if el.target == 0 else q1
            u1 = _rot(math.pi * el.amp / qp.pi_amp, el.phase_rad - frames[el.target])
            u = np.kron(u1, e) if el.target == 0 else np.kron(e, u1)
            r = u @ r @ u.conj().T
            r = _depolarize_1q_in_2q(r, qp.sq_depol, el.target)
            return free(r, int(el.duration_ps) - half), frames
        if isinstance(el, CZPulse):
            half = int(el.duration_ps // 2)
            r = free(r, half)
            phic = math.pi * el.amp / cz_true_amp
            f0, f1 = dyn_phase
            u = np.diag(np.exp(1j * np.array([0.0, f1, f0, f0 + f1 + phic])))
            r = u @ r @ u.conj().T
            r = _depolarize_2q(r, cz_depol)
            return free(r, int(el.duration_ps) - half), frames
        if isinstance(el, Repeat):
            if el.n <= 3 or (any(isinstance(x, (Drive, Repeat)) for x in el.body) and
                             any(isinstance(x, VirtualZ) for x in el.body)):
                for _ in range(el.n):
                    for x in el.body:
                        r, frames = step(x, r, frames)
                return r, frames
            d = r.shape[0]
            cols = []
            for k in range(d * d):
                ek = np.zeros(d * d, dtype=complex)
                ek[k] = 1.0
                out = ek.reshape(d, d, order="F")
                fr = frames
                for x in el.body:
                    out, fr = step(x, out, fr)
                cols.append(out.reshape(-1, order="F"))
            sup = np.column_stack(cols)
            r = (np.linalg.matrix_power(sup, el.n) @ r.reshape(-1, order="F")).reshape(d, d, order="F")
            f = list(frames)
            for x in el.body:
                if isinstance(x, VirtualZ):
                    f[x.target] += el.n * x.angle_rad
            return r, tuple(f)
        if isinstance(el, Twirled):
            def apply(r0, fr=frames):
                # inside a twirled block a virtual Z must act on the state itself
                # (there is no later pulse in the block to absorb the frame change):
                # VZ(theta) == RZ(theta) = diag(e^{-i theta/2}, e^{+i theta/2})
                for x in el.body:
                    if isinstance(x, VirtualZ):
                        rz = np.diag(np.exp(1j * np.array([-x.angle_rad / 2, x.angle_rad / 2])))
                        u = np.kron(rz, e) if x.target == 0 else np.kron(e, rz)
                        r0 = u @ r0 @ u.conj().T
                    else:
                        r0, _ = step(x, r0, fr)
                return r0
            sup = _block_super(apply, 4)
            tw = _twirl_super(sup, np.array(el.ideal, dtype=complex), 2, 4)
            r = (np.linalg.matrix_power(tw, el.n) @ r.reshape(-1, order="F")).reshape(4, 4, order="F")
            return r, frames
        raise TypeError(f"unknown element {el!r}")

    frames = (0.0, 0.0)
    for el in seq:
        rho, frames = step(el, rho, frames)
    pops = np.clip(np.real(np.diag(rho)), 0.0, None)
    return pops / pops.sum()

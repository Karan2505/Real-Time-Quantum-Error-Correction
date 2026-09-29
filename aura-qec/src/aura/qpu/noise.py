"""
Noise semantics shared by the virtual QPU (sampling) and the compiler
(detector-error-model generation).

Both sides call the *same* functions with a :class:`NoiseSnapshot`: the QPU
uses the true, drifting, control-dependent parameters; the compiler uses the
host's calibrated estimates.  When estimates equal the truth the compiler's
DEM is an exact description of what the QPU samples (verified in tests),
which is the property a decoder needs.

Channel placement rules (per operation on a qubit):

1. idle noise for the gap since the qubit's previous operation ended,
   applied immediately *before* the operation;
2. the ideal Clifford;
3. gate noise applied immediately *after* the operation;
4. measurement: the physical readout chain (QPU) / a classical record flip
   with the calibrated assignment error (DEM);
5. reset: X error after the reset.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import pauli as P


@dataclass
class QubitNoise:
    t1_ps: float
    t2_star_ps: float
    t2_echo_ps: float
    sq_depol: float = 0.0
    overrot_sx: float = 0.0       # fractional rotation-angle error of SX (pi/2) pulses
    overrot_x180: float = 0.0     # fractional rotation-angle error of X180 (pi) pulses
    detuning_hz: float = 0.0      # qubit frequency minus drive frame frequency
    meas_p01: float = 0.0         # P(read 1 | prepared 0)
    meas_p10: float = 0.0         # P(read 0 | prepared 1)
    reset_error: float = 0.0

    @property
    def meas_flip(self) -> float:
        return 0.5 * (self.meas_p01 + self.meas_p10)


@dataclass
class CouplerNoise:
    cz_depol: float = 0.0
    cz_phase_err_rad: float = 0.0         # conditional-phase error
    phase_err_a: float = 0.0              # residual single-qubit (dynamic) phase on first qubit
    phase_err_b: float = 0.0              # ... on second qubit
    spectator_z: float = 0.0

    def swapped(self) -> "CouplerNoise":
        return CouplerNoise(self.cz_depol, self.cz_phase_err_rad, self.phase_err_b,
                            self.phase_err_a, self.spectator_z)


@dataclass
class NoiseSnapshot:
    qubits: list[QubitNoise]
    couplers: dict[tuple[int, int], CouplerNoise]
    neighbors: list[list[int]]
    sx_ps: int
    x180_ps: int
    cz_ps: int

    def coupler(self, a: int, b: int) -> CouplerNoise:
        """Coupler noise oriented so phase_err_a refers to qubit ``a``."""
        c = self.couplers.get((a, b))
        if c is not None:
            return c
        c = self.couplers.get((b, a))
        if c is None:
            raise KeyError(f"no coupler between qubits {a} and {b}")
        return c.swapped()


# ----------------------------------------------------------------- channels
def idle_channel(qn: QubitNoise, gap_ps: float, echoed: bool = False, z_state: int = -1) -> np.ndarray:
    """Idle for ``gap_ps``.  ``echoed`` = protected by dynamical decoupling
    (T2echo instead of T2*, static detuning refocused).

    ``z_state``: if the qubit is known to sit in a computational-basis state
    (0 or 1), amplitude damping is applied exactly instead of Pauli-twirled:
    |1> relaxes (X error with probability 1 - exp(-t/T1)), |0> never gets
    excited; dephasing only contributes harmless Z.  -1 = general state (PTA).
    """
    if gap_ps <= 0:
        return np.array([1.0, 0.0, 0.0, 0.0])
    if z_state in (0, 1):
        t2 = min(qn.t2_echo_ps if echoed else qn.t2_star_ps, 2.0 * qn.t1_ps)
        gamma = 1.0 - math.exp(-gap_ps / qn.t1_ps)
        inv_phi = max(0.0, 1.0 / t2 - 0.5 / qn.t1_ps)
        pz = (1.0 - math.exp(-gap_ps * inv_phi)) / 2.0
        px = gamma if z_state == 1 else 0.0
        return P._check(np.array([0.0, px * (1 - pz), px * pz, (1 - px) * pz]))
    ch = P.pta_idle(qn.t1_ps, qn.t2_echo_ps if echoed else qn.t2_star_ps, gap_ps)
    if not echoed and qn.detuning_hz:
        pz = P.coherent_rotation_twirl(2 * math.pi * qn.detuning_hz * gap_ps * 1e-12)
        ch = P.compose(ch, P.pauli_channel_1(0.0, 0.0, pz))
    return ch


def sq_gate_channel(qn: QubitNoise, angle_rad: float, duration_ps: float, overrot: float) -> np.ndarray:
    """Noise after a single-qubit drive pulse of nominal rotation ``angle_rad``.

    Decoherence during the pulse + residual depolarising error + Pauli-twirled
    coherent over-rotation ``overrot`` (fractional angle error, along the drive
    axis; twirled to an X-type error)."""
    ch = P.pta_idle(qn.t1_ps, qn.t2_star_ps, duration_ps)
    if qn.sq_depol:
        ch = P.compose(ch, P.depolarize1(qn.sq_depol))
    if overrot:
        px = P.coherent_rotation_twirl(angle_rad * overrot)
        ch = P.compose(ch, P.pauli_channel_1(px, 0.0, 0.0))
    return ch


def cz_channel(qa: QubitNoise, qb: QubitNoise, cn: CouplerNoise, duration_ps: float) -> np.ndarray:
    """Two-qubit noise after a CZ: decoherence on both qubits during the pulse,
    depolarising gate error and twirled conditional-phase error."""
    ch = P.tensor(P.pta_idle(qa.t1_ps, qa.t2_star_ps, duration_ps),
                  P.pta_idle(qb.t1_ps, qb.t2_star_ps, duration_ps))
    if cn.cz_depol:
        ch = P.compose(ch, P.depolarize2(cn.cz_depol))
    if cn.cz_phase_err_rad:
        # exp(i d |11><11|) = exp(i d/4 (I - Z1 - Z2 + Z1 Z2)); twirl -> Z1, Z2, Z1Z2 each sin^2(d/4)
        p = math.sin(cn.cz_phase_err_rad / 4.0) ** 2
        v = np.zeros(16)
        v[3 * 4 + 0] = p     # Z I
        v[0 * 4 + 3] = p     # I Z
        v[3 * 4 + 3] = p     # Z Z
        ch = P.compose(ch, P._check(v))
    if cn.phase_err_a or cn.phase_err_b:
        za = P.coherent_rotation_twirl(cn.phase_err_a)
        zb = P.coherent_rotation_twirl(cn.phase_err_b)
        ch = P.compose(ch, P.tensor(P.pauli_channel_1(0, 0, za), P.pauli_channel_1(0, 0, zb)))
    return ch


# --------------------------------------------------------- noisy op stream
@dataclass
class NoiseLocation:
    """One stochastic Pauli channel at a point in the circuit."""
    qubits: tuple[int, ...]
    probs: np.ndarray              # length 4**len(qubits)
    tag: str = ""


@dataclass
class RecordFlip:
    """Classical flip of a measurement record (readout assignment error)."""
    qubit: int
    p: float
    tag: str = "readout"


def op_noise(kind: str, qubits: tuple[int, ...], snap: NoiseSnapshot) -> list[NoiseLocation]:
    """Gate noise placed *after* an operation (rule 3)."""
    out: list[NoiseLocation] = []
    if kind in ("SX", "X180"):
        q = qubits[0]
        qn = snap.qubits[q]
        if kind == "SX":
            ch = sq_gate_channel(qn, math.pi / 2, snap.sx_ps, qn.overrot_sx)
        else:
            ch = sq_gate_channel(qn, math.pi, snap.x180_ps, qn.overrot_x180)
        out.append(NoiseLocation((q,), ch, kind))
    elif kind == "CZ":
        a, b = qubits
        cn = snap.coupler(a, b)
        out.append(NoiseLocation((a, b), cz_channel(snap.qubits[a], snap.qubits[b], cn, snap.cz_ps), "CZ"))
        if cn.spectator_z:
            for s in sorted(set(snap.neighbors[a]) | set(snap.neighbors[b])):
                if s not in (a, b):
                    out.append(NoiseLocation((s,), P.pauli_channel_1(0, 0, cn.spectator_z), "spectator"))
    elif kind == "RESET":
        q = qubits[0]
        e = snap.qubits[q].reset_error
        if e:
            out.append(NoiseLocation((q,), P.pauli_channel_1(e, 0, 0), "reset"))
    return [loc for loc in out if loc.probs[1:].sum() > 0]


def build_snapshot(qubits: list[QubitNoise], couplers: dict[tuple[int, int], CouplerNoise],
                   neighbors: list[list[int]], sx_ps: int, x180_ps: int, cz_ps: int) -> NoiseSnapshot:
    return NoiseSnapshot(qubits=qubits, couplers=couplers, neighbors=neighbors,
                         sx_ps=sx_ps, x180_ps=x180_ps, cz_ps=cz_ps)

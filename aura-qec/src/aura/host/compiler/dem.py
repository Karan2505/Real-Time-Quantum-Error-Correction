"""
Detector error model (DEM) generation by backward sensitivity propagation.

For every detector / observable row we propagate its *sensitivity* (which
Pauli errors at the current point in the circuit would flip it) backwards
through the timed program, in the exact order the QPU applies operations
and noise (see ``qpu/noise.py``).  Sensitivities of all rows are packed into
bit-vectors, so one backward sweep over the program yields every error
mechanism:

    error E (frame bits ex, ez) at a point flips row R  <=>  ex.ax_R + ez.az_R = 1

Backward transposes of the forward frame maps:
    SX: az ^= ax            S: ax ^= az           H: swap
    CZ(a,b): ax_a ^= az_b ; ax_b ^= az_a
Measurement m of q:  rows containing m get ax_q ^= 1
Reset of q / measurement of q / circuit start: rows must have az_q = 0
    (their Z sensitivity would see the random post-measurement kick);
    this is the *detector determinism check*, and resets clear ax_q.

Disjoint Pauli channels are converted exactly into independent mechanisms
(``pauli.to_independent``); mechanisms with identical symptoms are merged
with XOR-probability addition.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from ...core.errors import CompileError
from ...qpu import noise as N
from ...qpu import pauli as P
from ...qpu.program import QpuProgram


@dataclass
class DetectorErrorModel:
    num_detectors: int
    num_observables: int
    probs: np.ndarray                           # (M,)
    dets: list[tuple[int, ...]]                 # detectors flipped by each mechanism
    obs: list[tuple[int, ...]]                  # observables flipped by each mechanism

    @property
    def num_mechanisms(self) -> int:
        return len(self.probs)

    def check_matrix(self) -> np.ndarray:
        h = np.zeros((self.num_detectors, self.num_mechanisms), dtype=np.uint8)
        for j, ds in enumerate(self.dets):
            h[list(ds), j] = 1
        return h

    def observable_matrix(self) -> np.ndarray:
        l = np.zeros((self.num_observables, self.num_mechanisms), dtype=np.uint8)
        for j, os_ in enumerate(self.obs):
            l[list(os_), j] = 1
        return l

    def detector_marginals(self) -> np.ndarray:
        """P(detector fires) assuming independent mechanisms."""
        prod = np.ones(self.num_detectors)
        for p, ds in zip(self.probs, self.dets):
            for d in ds:
                prod[d] *= (1.0 - 2.0 * p)
        return (1.0 - prod) / 2.0

    def as_dict(self) -> dict[tuple[tuple[int, ...], tuple[int, ...]], float]:
        return {(d, o): float(p) for p, d, o in zip(self.probs, self.dets, self.obs)}

    def restrict(self, detectors: Sequence[int]) -> "DetectorErrorModel":
        """Marginalise onto a subset of detectors (renumbered 0..len-1)."""
        remap = {d: i for i, d in enumerate(detectors)}
        acc: dict[tuple, float] = {}
        for p, ds, os_ in zip(self.probs, self.dets, self.obs):
            nd = tuple(sorted(remap[d] for d in ds if d in remap))
            if not nd and not os_:
                continue
            key = (nd, os_)
            q = acc.get(key, 0.0)
            acc[key] = q + p - 2 * q * p
        keys = sorted(acc)
        return DetectorErrorModel(len(detectors), self.num_observables, np.array([acc[k] for k in keys]),
                                  [k[0] for k in keys], [k[1] for k in keys])

    def to_json_dict(self) -> dict:
        return {"num_detectors": self.num_detectors, "num_observables": self.num_observables,
                "probs": [float(p) for p in self.probs], "dets": [list(d) for d in self.dets],
                "obs": [list(o) for o in self.obs]}

    @classmethod
    def from_json_dict(cls, d: dict) -> "DetectorErrorModel":
        return cls(d["num_detectors"], d["num_observables"], np.array(d["probs"], float),
                   [tuple(x) for x in d["dets"]], [tuple(x) for x in d["obs"]])


class DeterminismError(CompileError):
    pass


def generate_dem(program: QpuProgram, detectors: Sequence[Sequence[int]], observables: Sequence[Sequence[int]],
                 snap: Optional[N.NoiseSnapshot], meas_flip: Optional[Sequence[float]] = None,
                 detector_names: Optional[Sequence[str]] = None,
                 approximate_disjoint: bool = False,
                 z_states: Optional[Sequence[tuple]] = None) -> Optional[DetectorErrorModel]:
    """Build the DEM of ``program`` under noise ``snap`` (None -> determinism check only).

    ``detectors`` / ``observables``: lists of acquisition ids whose parity forms each row.
    ``meas_flip[q]``: classical readout flip probability per qubit (defaults to snap's).
    ``approximate_disjoint``: use q_P = p_P instead of the exact independent
    decomposition (matches Stim's ``approximate_disjoint_errors``).
    ``z_states``: per-op Z-eigenstate annotations (``qpu.reference.z_eigen_annotations``);
    idles of qubits in a computational-basis state use exact amplitude damping.
    """
    nd, no = len(detectors), len(observables)
    rows = nd + no
    nbytes = (rows + 7) // 8 or 1
    n = program.num_qubits

    def pack(idx: Sequence[int]) -> np.ndarray:
        v = np.zeros(rows, dtype=bool)
        v[list(idx)] = True
        return np.packbits(v)

    acq_rows: dict[int, list[int]] = {}
    for r, accs in enumerate(list(detectors) + list(observables)):
        for a in accs:
            acq_rows.setdefault(int(a), []).append(r)
    acq_vec = {a: pack(rs) for a, rs in acq_rows.items()}
    zero = np.zeros(nbytes, dtype=np.uint8)
    ax = np.zeros((n, nbytes), dtype=np.uint8)
    az = np.zeros((n, nbytes), dtype=np.uint8)
    mech: dict[bytes, float] = {}

    def add(vec: np.ndarray, p: float) -> None:
        if p <= 0 or not vec.any():
            return
        k = vec.tobytes()
        q = mech.get(k, 0.0)
        mech[k] = q + p - 2.0 * q * p

    def channel(qubits: tuple[int, ...], probs: np.ndarray) -> None:
        if snap is None:
            return
        if approximate_disjoint:
            indep = np.asarray(probs, float).copy()
            indep[0] = 0.0
        else:
            indep = P.to_independent(probs)
        k = len(qubits)
        for pidx in np.nonzero(indep)[0]:
            xs, zs = P.pauli_bits(int(pidx), k)
            v = zero.copy()
            for j, q in enumerate(qubits):
                if xs[j]:
                    v ^= ax[q]
                if zs[j]:
                    v ^= az[q]
            add(v, float(indep[pidx]))

    def require_z_clear(q: int, where: str) -> None:
        if az[q].any():
            bad = np.nonzero(np.unpackbits(az[q])[:rows])[0]
            names = [detector_names[b] if detector_names and b < nd else
                     (f"D{b}" if b < nd else f"L{b - nd}") for b in bad[:6]]
            raise DeterminismError(f"non-deterministic row(s) {names} (Z-sensitive at {where} of qubit {q})")

    # nominal timing, forward pass to find the idle gap before each op
    last_end = np.zeros(n, dtype=np.int64)
    gaps: list[list[int]] = []
    for o in program.ops:
        g = []
        for q in o.qubits:
            g.append(int(o.t_nominal_ps - last_end[q]))
            last_end[q] = o.t_nominal_ps + o.duration_ps
        gaps.append(g)

    flips = meas_flip
    if flips is None and snap is not None:
        flips = [qn.meas_flip for qn in snap.qubits]

    for o in reversed(program.ops):
        k = o.kind
        # ---- gate noise (after the op) ------------------------------------
        if snap is not None and k != "MEASURE":
            for loc in N.op_noise(k, o.qubits, snap):
                channel(loc.qubits, loc.probs)
        # ---- the op (transpose) -------------------------------------------
        if k == "SX":
            q = o.qubits[0]
            if o.phase_q % 2:        # forward: S, SX, S  -> backward: S^T, SX^T, S^T
                ax[q] ^= az[q]
                az[q] ^= ax[q]
                ax[q] ^= az[q]
            else:
                az[q] ^= ax[q]
        elif k == "X180":
            pass                     # Pauli (conditional ones are data-dependent, not noise)
        elif k == "CZ":
            a, b = o.qubits
            ax[a] ^= az[b]
            ax[b] ^= az[a]
        elif k == "MEASURE":
            q = o.qubits[0]
            require_z_clear(q, f"measurement op {o.index}")
            v = acq_vec.get(o.acq)
            if v is not None:
                if flips is not None and flips[q] > 0:
                    add(v, float(flips[q]))       # classical record flip
                ax[q] ^= v
        elif k == "RESET":
            q = o.qubits[0]
            require_z_clear(q, f"reset op {o.index}")
            ax[q] = 0
        # ---- idle noise (before the op) -------------------------------------
        if snap is not None:
            zs = z_states[o.index] if z_states is not None else (-1,) * len(o.qubits)
            for q, gap, z in zip(o.qubits, gaps[o.index], zs):
                if gap > 0:
                    channel((q,), N.idle_channel(snap.qubits[q], gap, o.echoed, z))
    for q in range(n):
        require_z_clear(q, "circuit start")
    if snap is None:
        return None
    keys = list(mech)
    probs = np.array([mech[k] for k in keys])
    dets, obs = [], []
    for k in keys:
        bits = np.unpackbits(np.frombuffer(k, dtype=np.uint8))[:rows]
        idx = np.nonzero(bits)[0]
        dets.append(tuple(int(i) for i in idx if i < nd))
        obs.append(tuple(int(i - nd) for i in idx if i >= nd))
    order = sorted(range(len(keys)), key=lambda j: (dets[j], obs[j]))
    return DetectorErrorModel(nd, no, probs[order], [dets[j] for j in order], [obs[j] for j in order])

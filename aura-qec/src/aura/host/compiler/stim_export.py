"""
Export a compiled, noisy program as a Stim circuit (interoperability and
independent cross-validation; Stim itself is *not* a runtime dependency).
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from ...qpu import noise as N
from ...qpu.program import QpuProgram


def _fmt(p: float) -> str:
    return repr(float(p))


def _pauli_channel(qubits: tuple[int, ...], probs: np.ndarray) -> str:
    if probs[1:].sum() <= 0:
        return ""
    if len(qubits) == 1:
        return f"PAULI_CHANNEL_1({_fmt(probs[1])}, {_fmt(probs[2])}, {_fmt(probs[3])}) {qubits[0]}\n"
    args = ", ".join(_fmt(x) for x in probs[1:])
    return f"PAULI_CHANNEL_2({args}) {qubits[0]} {qubits[1]}\n"


def to_stim_text(program: QpuProgram, detectors: Sequence[Sequence[int]], observables: Sequence[Sequence[int]],
                 snap: Optional[N.NoiseSnapshot], meas_flip: Optional[Sequence[float]] = None,
                 det_coords: Optional[Sequence[Sequence[float]]] = None,
                 z_states: Optional[Sequence[tuple]] = None) -> str:
    out = []
    n = program.num_qubits
    last_end = np.zeros(n, dtype=np.int64)
    flips = meas_flip
    if flips is None and snap is not None:
        flips = [qn.meas_flip for qn in snap.qubits]
    acq_rec: dict[int, int] = {}
    nrec = 0
    for o in program.ops:
        if o.conditional:
            raise ValueError("feed-forward programs cannot be exported to Stim")
        zs = z_states[o.index] if z_states is not None else (-1,) * len(o.qubits)
        for q, z in zip(o.qubits, zs):
            gap = int(o.t_nominal_ps - last_end[q])
            if snap is not None and gap > 0:
                out.append(_pauli_channel((q,), N.idle_channel(snap.qubits[q], gap, o.echoed, z)))
            last_end[q] = o.t_nominal_ps + o.duration_ps
        k = o.kind
        if k in ("SX", "X180"):
            q = o.qubits[0]
            ph = o.phase_q % 4
            if k == "X180":
                out.append(f"{'X' if ph % 2 == 0 else 'Y'} {q}\n")
            else:
                out.append(f"S_DAG {q}\n" * ph + f"SQRT_X {q}\n" + f"S {q}\n" * ph)
        elif k == "CZ":
            out.append(f"CZ {o.qubits[0]} {o.qubits[1]}\n")
        elif k == "MEASURE":
            q = o.qubits[0]
            p = flips[q] if flips is not None else 0.0
            out.append(f"M({_fmt(p)}) {q}\n" if p > 0 else f"M {q}\n")
            acq_rec[o.acq] = nrec
            nrec += 1
        elif k == "RESET":
            out.append(f"R {o.qubits[0]}\n")
        if snap is not None and k != "MEASURE":
            for loc in N.op_noise(k, o.qubits, snap):
                out.append(_pauli_channel(loc.qubits, loc.probs))
    for i, accs in enumerate(detectors):
        recs = " ".join(f"rec[{acq_rec[a] - nrec}]" for a in accs)
        coords = ""
        if det_coords is not None:
            coords = "(" + ", ".join(repr(float(c)) for c in det_coords[i]) + ")"
        out.append(f"DETECTOR{coords} {recs}\n")
    for i, accs in enumerate(observables):
        recs = " ".join(f"rec[{acq_rec[a] - nrec}]" for a in accs)
        out.append(f"OBSERVABLE_INCLUDE({i}) {recs}\n")
    return "".join(out)

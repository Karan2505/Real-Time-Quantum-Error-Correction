"""Noiseless reference execution of a QPU program (shared by QPU and compiler)."""

from __future__ import annotations

import numpy as np

from .program import QpuProgram
from .stabilizer import Tableau


def apply_drive(tab: Tableau, q: int, phase_q: int, pi: bool) -> None:
    """Drive pulse about axis phase_q * pi/2:  R = S^k . SX . S^-k  (or X / Y for pi pulses)."""
    k = phase_q % 4
    if pi:
        (tab.pauli_x if k % 2 == 0 else tab.pauli_y)([q])
        return
    for _ in range(k):
        tab.s_dag([q])
    tab.sx([q])
    for _ in range(k):
        tab.s([q])


def reference_sample(program: QpuProgram, rng: np.random.Generator) -> np.ndarray:
    """Measurement bits (indexed by op index; zero for non-measurement ops) of one
    noiseless execution.  Conditional ops are skipped (decision = 0)."""
    tab = Tableau(program.num_qubits, rng)
    ref = np.zeros(len(program.ops), dtype=bool)
    for o in program.ops:
        if o.conditional:
            continue
        if o.kind == "SX":
            apply_drive(tab, o.qubits[0], o.phase_q, pi=False)
        elif o.kind == "X180":
            apply_drive(tab, o.qubits[0], o.phase_q, pi=True)
        elif o.kind == "CZ":
            tab.cz([o.qubits[0]], [o.qubits[1]])
        elif o.kind == "MEASURE":
            ref[o.index] = bool(tab.measure_z(o.qubits[0])[0])
        elif o.kind == "RESET":
            tab.reset_z(o.qubits[0])
    return ref


def acq_bits(program: QpuProgram, ref_by_op: np.ndarray) -> np.ndarray:
    out = np.zeros(program.num_acq, dtype=bool)
    for o in program.ops:
        if o.kind == "MEASURE":
            out[o.acq] = ref_by_op[o.index]
    return out


def z_eigen_annotations(program: QpuProgram, ref_by_op: np.ndarray) -> list[tuple]:
    """For every op, the Z-eigenstate status of each of its qubits *just before* it,
    in the noiseless reference: a tuple aligned with ``op.qubits`` whose entries are
    the reference Z value (0/1) if the qubit is in a computational-basis state, else -1.

    Rules (exact for these native ops): reset -> |0>; measurement -> |ref bit>;
    X180 flips the value; CZ preserves each qubit's Z-eigenstatus (it commutes with Z);
    SX destroys it.  Conditional ops are absent from the reference and ignored.
    """
    state = [0] * program.num_qubits          # every qubit starts in |0>
    out: list[tuple] = []
    for o in program.ops:
        out.append(tuple(state[q] for q in o.qubits))
        if o.conditional:
            continue
        if o.kind == "RESET":
            state[o.qubits[0]] = 0
        elif o.kind == "MEASURE":
            state[o.qubits[0]] = int(ref_by_op[o.index])
        elif o.kind == "X180":
            q = o.qubits[0]
            if state[q] >= 0:
                state[q] ^= 1
        elif o.kind == "SX":
            state[o.qubits[0]] = -1
    return out

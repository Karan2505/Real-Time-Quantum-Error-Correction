"""
The physical operation list a QPU executes for one shot.

Only *physical* operations reach the qubits: drive pulses (SX / X180 with a
quarter-turn frame phase -- virtual Z gates live in the sequencer's phase
tracking), CZ flux pulses on couplers, measurements and resets.  The
compiler produces a :class:`QpuProgram` in canonical order (nominal start
time, then emission order), and the sequencers deliver the same ops at run
time, possibly delayed (e.g. while waiting for a decoder decision).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Optional

PHYSICAL_KINDS = ("SX", "X180", "CZ", "MEASURE", "RESET")


@dataclass(frozen=True)
class StaticOp:
    index: int                       # position in canonical order (unique id)
    kind: str                        # one of PHYSICAL_KINDS
    qubits: tuple[int, ...]
    duration_ps: int
    t_nominal_ps: int                # start time relative to shot start (no feedback delays)
    phase_q: int = 0                 # drive axis = phase_q * pi/2 (SX/X180)
    conditional: bool = False        # classically controlled (skipped in the reference)
    condition: str = ""              # name of the decision register it depends on
    acq: int = -1                    # acquisition id (MEASURE)
    echoed: bool = False             # the idle gap *before* this op is protected by DD
    coupler: int = -1                # coupler index for CZ

    def __post_init__(self) -> None:
        if self.kind not in PHYSICAL_KINDS:
            raise ValueError(f"unknown physical op kind {self.kind!r}")
        if self.kind == "CZ" and len(self.qubits) != 2:
            raise ValueError("CZ acts on two qubits")
        if self.kind != "CZ" and len(self.qubits) != 1:
            raise ValueError(f"{self.kind} acts on one qubit")
        if self.kind == "MEASURE" and self.acq < 0:
            raise ValueError("MEASURE needs an acquisition id")
        if self.duration_ps <= 0:
            raise ValueError("duration must be positive")


@dataclass
class QpuProgram:
    num_qubits: int
    ops: tuple[StaticOp, ...]
    num_acq: int
    shot_duration_ps: int
    name: str = "program"
    _fingerprint: Optional[str] = field(default=None, repr=False)

    def fingerprint(self) -> str:
        if self._fingerprint is None:
            h = hashlib.blake2b(digest_size=16)
            h.update(str(self.num_qubits).encode())
            for o in self.ops:
                h.update(repr((o.index, o.kind, o.qubits, o.duration_ps, o.t_nominal_ps, o.phase_q,
                               o.conditional, o.acq, o.echoed)).encode())
            self._fingerprint = h.hexdigest()
        return self._fingerprint

    def validate(self) -> None:
        seen_acq = set()
        busy = [0] * self.num_qubits
        last_t = -1
        for i, o in enumerate(self.ops):
            if o.index != i:
                raise ValueError(f"op {i} has index {o.index}")
            if o.t_nominal_ps < last_t:
                raise ValueError("ops are not in canonical (time) order")
            last_t = o.t_nominal_ps
            for q in o.qubits:
                if not 0 <= q < self.num_qubits:
                    raise ValueError(f"op {i}: qubit {q} out of range")
                if o.t_nominal_ps < busy[q]:
                    raise ValueError(f"op {i} ({o.kind}) overlaps previous op on qubit {q}")
                busy[q] = o.t_nominal_ps + o.duration_ps
            if o.kind == "MEASURE":
                if o.acq in seen_acq:
                    raise ValueError(f"acquisition {o.acq} used twice")
                seen_acq.add(o.acq)
        if seen_acq != set(range(self.num_acq)):
            raise ValueError("acquisition ids are not 0..num_acq-1")

"""
Logical syndrome-extraction circuit builder.

Produces an ordered list of logical operations (reset, H, CX, measure) on
*device* qubit indices, grouped into moments, plus the measurement keys
needed to build detectors.  Two scheduling strategies:

* ``interleaved`` -- X and Z checks measured simultaneously using the
  code's hook-aware CNOT order (rotated surface / repetition codes);
* ``separated`` -- all Z checks, then all X checks, each half scheduled with
  an optimal (max-degree) bipartite edge colouring.  Valid for *any* CSS
  code (the two halves commute), used for BB codes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ...codes.css import CSSCode
from ...core.errors import CompileError
from ...qpu.device import DeviceSpec
from .experiment import ExperimentSpec, PatchSpec


@dataclass(frozen=True)
class LOp:
    kind: str                         # 'R' | 'H' | 'CX' | 'M' | 'CONDX'
    qubits: tuple[int, ...]
    moment: int
    key: tuple = ()                   # M: (patch, round, role, index); role in 'x','z','d','flag'
    barrier: str = ""                 # M ops sharing a barrier start readout together


def bipartite_edge_coloring(edges: list[tuple[int, int]]) -> list[int]:
    """Proper edge colouring of a bipartite (multi)graph with max-degree colours
    (Konig).  ``edges`` are (left, right) pairs.  Returns a colour per edge."""
    color = [-1] * len(edges)
    at: dict[tuple[str, int], dict[int, int]] = {}

    def slot(node):
        return at.setdefault(node, {})

    def free(node) -> int:
        s = slot(node)
        c = 0
        while c in s:
            c += 1
        return c

    for e, (u, v) in enumerate(edges):
        U, V = ("L", u), ("R", v)
        a, b = free(U), free(V)
        if a not in slot(V):
            c = a
        else:
            # flip the a/b alternating path starting at V (it cannot reach U)
            path = []
            node, col = V, a
            while col in slot(node):
                pe = slot(node)[col]
                path.append(pe)
                pu, pv = edges[pe]
                node = ("L", pu) if node == ("R", pv) else ("R", pv)
                col = b if col == a else a
            for pe in path:
                pu, pv = edges[pe]
                old = color[pe]
                del slot(("L", pu))[old]
                del slot(("R", pv))[old]
            for pe in path:
                pu, pv = edges[pe]
                new = b if color[pe] == a else a
                color[pe] = new
                slot(("L", pu))[new] = pe
                slot(("R", pv))[new] = pe
            c = a
        color[e] = c
        slot(U)[c] = e
        slot(V)[c] = e
    return color


def _separated_steps(h: np.ndarray) -> tuple[tuple[int, ...], ...]:
    edges = [(int(c), int(q)) for c, q in zip(*np.nonzero(h))]
    if not edges:
        return tuple(() for _ in range(h.shape[0]))
    cols = bipartite_edge_coloring(edges)
    depth = max(cols) + 1
    steps = [[-1] * depth for _ in range(h.shape[0])]
    for (c, q), col in zip(edges, cols):
        steps[c][col] = q
    return tuple(tuple(r) for r in steps)


@dataclass
class PatchLayout:
    patch: PatchSpec
    data: list[int]
    x_anc: list[int]
    z_anc: list[int]


@dataclass
class LogicalCircuit:
    ops: list[LOp]
    layouts: list[PatchLayout]
    schedule_used: dict[int, str] = field(default_factory=dict)
    flag_qubit: Optional[int] = None


def patch_layout(spec: DeviceSpec, p: PatchSpec) -> PatchLayout:
    by_name = spec.qubit_by_name()
    try:
        data = [by_name[f"p{p.patch_id}.d{i}"] for i in range(p.code.n)]
        xa = [by_name[f"p{p.patch_id}.x{c}"] for c in range(p.code.num_x_checks)]
        za = [by_name[f"p{p.patch_id}.z{c}"] for c in range(p.code.num_z_checks)]
    except KeyError as e:
        raise CompileError(f"device {spec.name} has no qubit {e} for patch {p.patch_id}") from None
    return PatchLayout(p, data, xa, za)


def build_logical_circuit(spec: DeviceSpec, exp: ExperimentSpec) -> LogicalCircuit:
    exp.validate()
    ops: list[LOp] = []
    layouts = [patch_layout(spec, p) for p in exp.patches]
    sched_used: dict[int, str] = {}
    moment = 0

    per_patch_moments: list[list[list[LOp]]] = []
    for lay in layouts:
        p = lay.patch
        code: CSSCode = p.code
        mode = exp.options.schedule
        if mode == "auto":
            mode = "interleaved" if code.schedule is not None else "separated"
        if mode == "interleaved" and code.schedule is None:
            raise CompileError(f"code {code.name} has no interleaved schedule; use 'separated'")
        sched_used[p.patch_id] = mode
        if mode == "interleaved":
            xs, zs = code.schedule.x_steps, code.schedule.z_steps
            layers = [[] for _ in range(code.schedule.depth)]
            for t in range(code.schedule.depth):
                for c, row in enumerate(xs):
                    if row[t] >= 0:
                        layers[t].append(("x", c, row[t]))
                for c, row in enumerate(zs):
                    if row[t] >= 0:
                        layers[t].append(("z", c, row[t]))
        else:
            zsteps = _separated_steps(code.hz) if code.num_z_checks else ()
            xsteps = _separated_steps(code.hx) if code.num_x_checks else ()
            zd = len(zsteps[0]) if zsteps else 0
            xd = len(xsteps[0]) if xsteps else 0
            layers = [[] for _ in range(zd + xd)]
            for c, row in enumerate(zsteps):
                for t, q in enumerate(row):
                    if q >= 0:
                        layers[t].append(("z", c, q))
            for c, row in enumerate(xsteps):
                for t, q in enumerate(row):
                    if q >= 0:
                        layers[zd + t].append(("x", c, q))

        moments: list[list[LOp]] = []
        pid = p.patch_id
        allq = lay.data + lay.x_anc + lay.z_anc
        moments.append([LOp("R", (q,), 0) for q in allq])
        if p.basis == "X":
            moments.append([LOp("H", (q,), 0) for q in lay.data])
        for r in range(p.rounds):
            if lay.x_anc:
                moments.append([LOp("H", (q,), 0) for q in lay.x_anc])
            for layer in layers:
                m = []
                for role, c, dq in layer:
                    d = lay.data[dq]
                    if role == "x":
                        m.append(LOp("CX", (lay.x_anc[c], d), 0))
                    else:
                        m.append(LOp("CX", (d, lay.z_anc[c]), 0))
                moments.append(m)
            if lay.x_anc:
                moments.append([LOp("H", (q,), 0) for q in lay.x_anc])
            bar = f"p{pid}.r{r}"
            meas = [LOp("M", (q,), 0, (pid, r, "x", c), bar) for c, q in enumerate(lay.x_anc)]
            meas += [LOp("M", (q,), 0, (pid, r, "z", c), bar) for c, q in enumerate(lay.z_anc)]
            moments.append(meas)
            moments.append([LOp("R", (q,), 0) for q in lay.x_anc + lay.z_anc])
        if p.basis == "X":
            moments.append([LOp("H", (q,), 0) for q in lay.data])
        bar = f"p{pid}.final"
        moments.append([LOp("M", (q,), 0, (pid, p.rounds, "d", i), bar) for i, q in enumerate(lay.data)])
        per_patch_moments.append(moments)

    # merge patches moment-by-moment (disjoint qubits run in parallel)
    depth = max(len(m) for m in per_patch_moments)
    for t in range(depth):
        for mlist in per_patch_moments:
            if t < len(mlist):
                for o in mlist[t]:
                    ops.append(LOp(o.kind, o.qubits, moment, o.key, o.barrier))
        moment += 1

    flag = None
    if exp.options.branching:
        # feed-forward demo: after the experiment, the first X- or Z-ancilla of the first
        # patch is conditionally flipped by the decoder's logical decision and measured.
        lay = layouts[0]
        cands = lay.z_anc + lay.x_anc
        if not cands:
            raise CompileError("branching needs an ancilla qubit to use as flag")
        flag = cands[0]
        ops.append(LOp("R", (flag,), moment)); moment += 1
        ops.append(LOp("CONDX", (flag,), moment, key=("decision", lay.patch.patch_id))); moment += 1
        ops.append(LOp("M", (flag,), moment, (lay.patch.patch_id, lay.patch.rounds + 1, "flag", 0),
                       f"p{lay.patch.patch_id}.flag")); moment += 1
    return LogicalCircuit(ops, layouts, sched_used, flag)

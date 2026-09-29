"""
Calibration graph (dependency-aware maintenance, in the spirit of Kelly et
al., "Physical qubit calibration on a directed acyclic graph", 2018).

Each node knows its dependencies, how long its result stays valid, an
optional cheap *check* (a short experiment that says whether the stored value
is still in spec for each qubit) and the full *calibration*.  ``maintain``
walks the graph in dependency order and only recalibrates what is stale or
out of spec -- so routine maintenance costs a fraction of a full bring-up.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from ...core.errors import CalibrationError
from ...core.units import S
from .experiments import Calibrator


@dataclass
class CalNode:
    name: str
    deps: tuple[str, ...]
    scope: str                                   # 'qubit' | 'coupler'
    run: Callable[[Calibrator, Optional[list[int]]], dict]
    timeout_s: float                             # validity (lab seconds)
    check: Optional[Callable[[Calibrator, list[int]], list[int]]] = None   # returns out-of-spec targets
    group: str = ""                              # store update group used for staleness
    cascade_from: tuple[str, ...] = ()           # deps whose recalibration invalidates this node


def _rabi_check(cal: Calibrator, qs: list[int], tol: float = 2e-3) -> list[int]:
    r = cal.rabi_fine(qs, check_only=True)
    return [q for q in qs if abs(r[q]["eps"]) > tol]


def _ramsey_check(cal: Calibrator, qs: list[int], tol_hz: float = 20e3) -> list[int]:
    r = cal.ramsey(qs, check_only=True)
    return [q for q in qs if abs(r[q]["detuning_hz"]) > tol_hz]


DEFAULT_NODES: list[CalNode] = [
    CalNode("readout_coarse", (), "qubit", lambda c, t: c.readout(t), 3600, group="readout"),
    CalNode("rabi", ("readout_coarse",), "qubit", lambda c, t: c.rabi(t), 3600, group="rabi"),
    CalNode("ramsey", ("rabi",), "qubit", lambda c, t: c.ramsey(t), 600, _ramsey_check, group="ramsey"),
    CalNode("rabi_fine", ("ramsey",), "qubit", lambda c, t: c.rabi_fine(t), 1800, _rabi_check, group="rabi_fine",
            cascade_from=("ramsey",)),
    CalNode("readout", ("rabi_fine",), "qubit", lambda c, t: c.readout(t), 1800, group="readout"),
    CalNode("t1", ("readout",), "qubit", lambda c, t: c.t1(t), 3600, group="t1"),
    CalNode("t2echo", ("t1",), "qubit", lambda c, t: c.t2echo(t), 3600, group="t2echo"),
    CalNode("sq_bench", ("t1",), "qubit", lambda c, t: c.sq_bench(t), 3600, group="sq_bench"),
    CalNode("cz_phase", ("readout", "ramsey"), "coupler", lambda c, t: c.cz_phase(t), 1800, group="cz_phase"),
    CalNode("cz_bench", ("cz_phase", "t1"), "coupler", lambda c, t: c.cz_bench(t), 3600, group="cz_bench"),
]


@dataclass
class CalReport:
    actions: list[tuple[str, str, list[int]]] = field(default_factory=list)   # (node, action, targets)
    virtual_time_ps: int = 0

    def summary(self) -> str:
        return "; ".join(f"{n}:{a}({len(t)})" for n, a, t in self.actions)


class CalibrationGraph:
    def __init__(self, calibrator: Calibrator, nodes: Optional[list[CalNode]] = None):
        self.cal = calibrator
        self.nodes = {n.name: n for n in (nodes or DEFAULT_NODES)}
        self._order = self._toposort()

    def _toposort(self) -> list[str]:
        order, seen, tmp = [], set(), set()

        def visit(n: str) -> None:
            if n in seen:
                return
            if n in tmp:
                raise CalibrationError(f"cycle in calibration graph at {n}")
            tmp.add(n)
            for d in self.nodes[n].deps:
                if d not in self.nodes:
                    raise CalibrationError(f"node {n} depends on unknown node {d}")
                visit(d)
            tmp.discard(n)
            seen.add(n)
            order.append(n)
        for n in self.nodes:
            visit(n)
        return order

    def _targets(self, node: CalNode) -> list[int]:
        spec = self.cal.spec
        return list(range(spec.num_qubits)) if node.scope == "qubit" else list(range(len(spec.couplers)))

    def _lab_age_s(self, node: CalNode, target: int) -> float:
        st = self.cal.store
        now = self.cal.sim.now
        age_ps = st.age_ps(now, q=target, group=node.group) if node.scope == "qubit" \
            else st.age_ps(now, coupler=target, group=node.group)
        speed = self.cal.qpu.dev.drift.speedup
        return age_ps * speed / S

    def calibrate_all(self) -> CalReport:
        """Full bring-up: every node, every target, in dependency order."""
        rep = CalReport()
        t0 = self.cal.sim.now
        for name in self._order:
            node = self.nodes[name]
            tg = self._targets(node)
            node.run(self.cal, None)
            rep.actions.append((name, "calibrate", tg))
        rep.virtual_time_ps = self.cal.sim.now - t0
        return rep

    def maintain(self) -> CalReport:
        """Recalibrate only stale / out-of-spec targets (and their dependents)."""
        rep = CalReport()
        t0 = self.cal.sim.now
        dirty: dict[str, set[int]] = {}
        for name in self._order:
            node = self.nodes[name]
            tg = self._targets(node)
            need = {t for t in tg if self._lab_age_s(node, t) > node.timeout_s}
            # only edges that genuinely couple parameters cascade (e.g. a new drive frequency
            # shifts the optimal pulse amplitude); everything else is covered by the node's own
            # staleness timeout and in-spec check
            for d in node.cascade_from:
                if d in dirty and dirty[d]:
                    if node.scope == self.nodes[d].scope:
                        need |= dirty[d]
                    elif node.scope == "coupler":
                        qs = dirty[d]
                        need |= {c.index for c in self.cal.spec.couplers if c.q0 in qs or c.q1 in qs}
            if node.check is not None and node.scope == "qubit":
                fresh = [t for t in tg if t not in need]
                if fresh:
                    bad = node.check(self.cal, fresh)
                    rep.actions.append((name, "check", fresh))
                    need |= set(bad)
            if need:
                node.run(self.cal, sorted(need))
                rep.actions.append((name, "calibrate", sorted(need)))
            dirty[name] = need
        rep.virtual_time_ps = self.cal.sim.now - t0
        return rep

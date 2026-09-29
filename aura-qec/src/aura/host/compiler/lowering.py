"""
Lowering: logical circuit -> native gates -> timed schedule -> QPU program
and per-channel sequencer programs.

Native gate set of the virtual chip: SX (pi/2 drive), X180, virtual Z
(frame update, free), CZ (coupler flux pulse), MEASURE, RESET.

* CX(c, t) = H(t) CZ(c, t) H(t), with adjacent H.H pairs cancelled per qubit.
* H = VZ(pi/2) SX VZ(pi/2): virtual Zs are folded into the drive-frame phase
  of later pulses (``phase_q``); the residual Z rotation at the end commutes
  with CZ and Z-basis measurement, so no physical pulse is needed.
* Timing is per-qubit ASAP on the sequencer grid; all measurements of a
  round start together (one readout trigger per round).
* Dynamical decoupling (X180 pairs) is inserted into long idle windows of
  data qubits (typically while the ancillas are read out).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ...control.isa import ChannelProgram, Instr, SequencerBundle
from ...core.errors import CompileError
from ...core.units import ceil_to_grid
from ...qpu.device import DeviceSpec
from ...qpu.program import QpuProgram, StaticOp
from .builder import LOp, LogicalCircuit
from .experiment import CompileOptions


@dataclass
class NOp:
    """A native, timed operation (before canonical indexing)."""
    kind: str                     # 'VZ','SX','X180','CZ','MEASURE','RESET'
    qubits: tuple[int, ...]
    seq: int                      # program order (tie breaker)
    quarter: int = 0              # VZ amount
    phase_q: int = 0
    t: int = -1
    duration: int = 0
    key: tuple = ()
    barrier: str = ""
    conditional: bool = False
    condition: str = ""
    echoed: bool = False
    dd: bool = False


@dataclass
class LoweredProgram:
    qpu: QpuProgram
    sequencers: SequencerBundle
    acq_keys: list[tuple]                 # acq id -> measurement key
    acq_qubit: list[int]
    acq_time_ps: list[int]                # nominal readout start per acq
    round_end_ps: dict[tuple[int, int], int]   # (patch, round) -> nominal time the last readout ends
    stats: dict = field(default_factory=dict)


def _expand_native(ops: list[LOp]) -> list[NOp]:
    """CX -> H CZ H, then cancel adjacent H pairs per qubit, then H -> VZ SX VZ."""
    # 1. expand CX moment-by-moment into three sub-moments
    out: list[tuple[str, tuple[int, ...], LOp]] = []
    i = 0
    while i < len(ops):
        m = ops[i].moment
        j = i
        while j < len(ops) and ops[j].moment == m:
            j += 1
        group = ops[i:j]
        used = [q for o in group for q in o.qubits]
        if len(used) != len(set(used)):
            raise CompileError(f"moment {m}: a qubit is used by two operations")
        # ops of one moment act on disjoint qubits (e.g. patch A's CX layer next to
        # patch B's H layer), so their relative order is irrelevant
        cxs = [o for o in group if o.kind == "CX"]
        out += [(o.kind, o.qubits, o) for o in group if o.kind != "CX"]
        if cxs:
            out += [("H", (o.qubits[1],), o) for o in cxs]
            out += [("CZ", o.qubits, o) for o in cxs]
            out += [("H", (o.qubits[1],), o) for o in cxs]
        i = j
    # 2. cancel H.H on the same qubit when adjacent in that qubit's sequence
    keep = [True] * len(out)
    stack: dict[int, list[int]] = {}
    for idx, (k, qs, _) in enumerate(out):
        for q in qs:
            st = stack.setdefault(q, [])
            if k == "H" and st and out[st[-1]][0] == "H" and keep[st[-1]]:
                keep[st[-1]] = False
                keep[idx] = False
                st.pop()
            else:
                st.append(idx)
    # 3. native conversion with frame-phase tracking
    phase: dict[int, int] = {}
    native: list[NOp] = []
    seq = 0

    def add(**kw):
        nonlocal seq
        native.append(NOp(seq=seq, **kw))
        seq += 1

    for idx, (k, qs, lop) in enumerate(out):
        if not keep[idx]:
            continue
        if k == "H":
            q = qs[0]
            # H = RZ(pi/2) . SX . RZ(pi/2): a VZ(theta) shifts later pulse axes by -theta
            add(kind="VZ", qubits=qs, quarter=1)
            phase[q] = (phase.get(q, 0) - 1) % 4
            add(kind="SX", qubits=qs, phase_q=phase[q])
            add(kind="VZ", qubits=qs, quarter=1)
            phase[q] = (phase.get(q, 0) - 1) % 4
        elif k == "CZ":
            add(kind="CZ", qubits=qs)
        elif k == "R":
            phase[qs[0]] = 0
            add(kind="RESET", qubits=qs)
        elif k == "M":
            add(kind="MEASURE", qubits=qs, key=lop.key, barrier=lop.barrier)
        elif k == "CONDX":
            add(kind="X180", qubits=qs, phase_q=phase.get(qs[0], 0), conditional=True,
                condition=f"L{lop.key[1]}")
        else:
            raise CompileError(f"cannot lower op {k}")
    return native


def _duration(spec: DeviceSpec, kind: str) -> int:
    d = spec.durations
    return {"VZ": 0, "SX": d.sx_ps, "X180": d.x180_ps, "CZ": d.cz_ps, "MEASURE": d.meas_ps,
            "RESET": d.reset_ps}[kind]


def _schedule(spec: DeviceSpec, native: list[NOp], opts: CompileOptions) -> None:
    grid = spec.durations.grid_ps
    avail = [0] * spec.num_qubits
    i = 0
    while i < len(native):
        o = native[i]
        if o.kind == "MEASURE" and o.barrier and opts.measurement_barrier:
            j = i
            while j < len(native) and native[j].kind == "MEASURE" and native[j].barrier == o.barrier:
                j += 1
            group = native[i:j]
            t = ceil_to_grid(max(avail[g.qubits[0]] for g in group), grid)
            for g in group:
                g.t = t
                g.duration = _duration(spec, "MEASURE")
                avail[g.qubits[0]] = t + g.duration
            i = j
            continue
        o.duration = _duration(spec, o.kind)
        o.t = ceil_to_grid(max(avail[q] for q in o.qubits), grid)
        if o.kind != "VZ":
            for q in o.qubits:
                avail[q] = o.t + o.duration
        i += 1


def _insert_dd(spec: DeviceSpec, native: list[NOp], opts: CompileOptions, data_qubits: set[int]) -> list[NOp]:
    grid = spec.durations.grid_ps
    x = spec.durations.x180_ps
    per_q: dict[int, list[NOp]] = {}
    for o in native:
        if o.kind == "VZ":
            continue
        for q in o.qubits:
            per_q.setdefault(q, []).append(o)
    extra: list[NOp] = []
    for q, lst in per_q.items():
        if q not in data_qubits:
            continue
        lst.sort(key=lambda o: (o.t, o.seq))
        for a, b in zip(lst, lst[1:]):
            start, end = a.t + a.duration, b.t
            gap = end - start
            if gap < max(opts.dd_min_idle_ps, 4 * x):
                continue
            t1 = ceil_to_grid(start + gap // 4 - x // 2, grid)
            t2 = ceil_to_grid(start + (3 * gap) // 4 - x // 2, grid)
            if t1 < start or t2 < t1 + x or t2 + x > end:
                continue
            extra.append(NOp(kind="X180", qubits=(q,), seq=a.seq, t=t1, duration=x, echoed=True, dd=True))
            extra.append(NOp(kind="X180", qubits=(q,), seq=a.seq, t=t2, duration=x, echoed=True, dd=True))
            b.echoed = True
    out = native + extra
    _assign_frame_phases(out)
    return out


def _assign_frame_phases(native: list[NOp]) -> None:
    """Re-derive every drive pulse's frame phase by replaying the per-qubit
    time-ordered stream of virtual Zs and resets (needed for inserted pulses)."""
    per_q: dict[int, list[NOp]] = {}
    for o in native:
        if o.kind in ("VZ", "SX", "X180", "RESET"):
            per_q.setdefault(o.qubits[0], []).append(o)
    for q, lst in per_q.items():
        lst.sort(key=lambda o: (o.t, o.seq, 0 if o.kind == "VZ" else 1))
        ph = 0
        for o in lst:
            if o.kind == "VZ":
                ph = (ph - o.quarter) % 4
            elif o.kind == "RESET":
                ph = 0
            else:
                if not o.dd and o.phase_q % 4 != ph:
                    raise CompileError(f"frame-phase inconsistency on qubit {q} at t={o.t}")
                o.phase_q = ph


def lower(spec: DeviceSpec, circ: LogicalCircuit, opts: CompileOptions, name: str = "program") -> LoweredProgram:
    native = _expand_native(circ.ops)
    _schedule(spec, native, opts)
    data = {q for lay in circ.layouts for q in lay.data}
    if opts.dynamical_decoupling:
        native = _insert_dd(spec, native, opts, data)
    phys = sorted([o for o in native if o.kind != "VZ"], key=lambda o: (o.t, o.seq, o.qubits))
    cidx = spec.coupler_index()
    static: list[StaticOp] = []
    acq_keys: list[tuple] = []
    acq_qubit: list[int] = []
    acq_time: list[int] = []
    op_of: dict[int, int] = {}
    for o in phys:
        acq = -1
        if o.kind == "MEASURE":
            acq = len(acq_keys)
            acq_keys.append(o.key)
            acq_qubit.append(o.qubits[0])
            acq_time.append(o.t)
        cpl = -1
        if o.kind == "CZ":
            if (o.qubits[0], o.qubits[1]) not in cidx:
                raise CompileError(f"no coupler for CZ on qubits {o.qubits}")
            cpl = cidx[(o.qubits[0], o.qubits[1])]
        so = StaticOp(index=len(static), kind=o.kind, qubits=o.qubits, duration_ps=o.duration,
                      t_nominal_ps=o.t, phase_q=o.phase_q % 4, conditional=o.conditional,
                      condition=o.condition, acq=acq, echoed=o.echoed, coupler=cpl)
        op_of[id(o)] = so.index
        static.append(so)
    end = max((s.t_nominal_ps + s.duration_ps for s in static), default=0)
    prog = QpuProgram(num_qubits=spec.num_qubits, ops=tuple(static), num_acq=len(acq_keys),
                      shot_duration_ps=end, name=name)
    prog.validate()

    # ---- sequencer programs ------------------------------------------------------
    grid = spec.durations.grid_ps
    q_events: dict[int, list[tuple[int, int, NOp]]] = {q: [] for q in range(spec.num_qubits)}
    c_events: dict[int, list[tuple[int, int, NOp]]] = {}
    for o in native:
        if o.kind == "CZ":
            c_events.setdefault(cidx[(o.qubits[0], o.qubits[1])], []).append((o.t, o.seq, o))
        else:
            q_events[o.qubits[0]].append((o.t, o.seq, o))
    channels: list[ChannelProgram] = []

    def build(evs: list[tuple[int, int, NOp]]) -> tuple[Instr, ...]:
        evs.sort(key=lambda e: (e[0], e[1]))
        instrs: list[Instr] = []
        now = 0
        for t, _, o in evs:
            if t < now:
                raise CompileError(f"channel overlap at {t} ps (busy until {now})")
            if t > now:
                if (t - now) % grid:
                    raise CompileError("timing not on grid")
                instrs.append(Instr("WAIT", cycles=(t - now) // grid))
                now = t
            if o.kind == "VZ":
                instrs.append(Instr("VZ", quarter=o.quarter))
                continue
            idx = op_of[id(o)]
            if o.conditional:
                instrs.append(Instr("WAIT_DECISION", register=o.condition,
                                    cycles=max(1, opts.decision_timeout_ps // grid)))
                instrs.append(Instr("COND_PLAY", register=o.condition, op_index=idx, quarter=o.phase_q % 4))
            elif o.kind == "MEASURE":
                instrs.append(Instr("ACQUIRE", op_index=idx))
            elif o.kind == "RESET":
                instrs.append(Instr("RESET", op_index=idx))
            else:
                instrs.append(Instr("PLAY", op_index=idx, quarter=o.phase_q % 4))
            now = t + o.duration
        instrs.append(Instr("END"))
        return tuple(instrs)

    for q in range(spec.num_qubits):
        if q_events[q]:
            channels.append(ChannelProgram(f"q{q}", "qubit", q, build(q_events[q])))
    for c in sorted(c_events):
        channels.append(ChannelProgram(f"c{c}", "coupler", c, build(c_events[c])))
    bundle = SequencerBundle(channels=tuple(channels), barriers={}, grid_ps=grid)

    round_end: dict[tuple[int, int], int] = {}
    for key, t in zip(acq_keys, acq_time):
        pr = (key[0], key[1])
        round_end[pr] = max(round_end.get(pr, 0), t + spec.durations.meas_ps)
    counts: dict[str, int] = {}
    for s in static:
        counts[s.kind] = counts.get(s.kind, 0) + 1
    stats = {"ops": counts, "shot_duration_ns": end / 1000.0, "dd_pulses": sum(1 for o in native if o.dd),
             "channels": len(channels)}
    return LoweredProgram(prog, bundle, acq_keys, acq_qubit, acq_time, round_end, stats)

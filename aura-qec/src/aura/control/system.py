"""
The virtual real-time control system.

Architecture (mirrors modern superconducting control stacks):

  timing controller --shot trigger--> per-channel SEQUENCERS --pulse bus--> QPU
                                                                             |
  QPU readout --IQ--> READOUT CARDS (discriminate, LLR, leakage flag, timestamp)
                          |
                          v
                 low-latency RESULTS CROSSBAR --+--> SYNDROME PACKETIZER --> AURA-Link TX
                                                +--> DATA CAPTURE (host)
  AURA-Link RX --corrections--> PAULI FRAME TRACKER --decision registers--> SEQUENCERS
                                                                        (feed-forward)

All latencies are explicit, configurable and applied in virtual time; the
defaults are of the same order as the component breakdown published for a
real FPGA control system with an integrated decoder (Caune et al., Nature
Communications 17, 7383 (2026)).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Protocol

import numpy as np

from ..core.clock import LocalClock, TrueTime
from ..core.errors import HardwareFault
from ..core.rng import RngFactory
from ..core.sim import Simulator
from ..core.telemetry import MetricsRegistry
from ..core.units import NS
from ..link.records import FLAG_FIRST, FLAG_LAST, FLAG_LEAK, FLAG_SOFT, CorrectionRecord, SyndromeRecord
from ..qpu.program import QpuProgram, StaticOp
from ..qpu.qpu import VirtualQPU
from .isa import ChannelProgram, Instr
from .readout import Discriminator, quantize_llr


@dataclass
class ControlTiming:
    readout_chain_ps: int = 400 * NS      # qubit -> resonator -> amplifiers -> ADC pipeline
    classify_ps: int = 120 * NS           # integrate tail + discriminate + serialise
    crossbar_ps: int = 140 * NS           # low-latency results crossbar hop
    packetize_ps: int = 40 * NS           # syndrome-frame assembly
    feedback_route_ps: int = 160 * NS     # decision register broadcast to sequencers
    soft_bits: int = 4                    # 0, 4 or 8
    llr_step: float = 1.0                 # LLR quantisation step (nats per LSB)
    leak_flags: bool = True
    clock_offset_ps: int = 0
    clock_ppm: float = 0.0
    clock_resolution_ps: int = 1000       # 1 ns timestamps
    shot_start_poll_ps: int = 1000 * NS   # retry interval when the link applies shot-boundary backpressure

    def validate(self) -> None:
        if self.soft_bits not in (0, 4, 8):
            raise ValueError("soft_bits must be 0, 4 or 8")
        for f in ("readout_chain_ps", "classify_ps", "crossbar_ps", "packetize_ps", "feedback_route_ps"):
            if getattr(self, f) < 0:
                raise ValueError(f"{f} must be >= 0")


class SyndromeSink(Protocol):
    def send_syndrome(self, rec: SyndromeRecord) -> None: ...

    def can_start_shot(self) -> bool: ...


@dataclass
class ShotRecord:
    shot: int
    t_start: int
    bits: np.ndarray
    llr: np.ndarray
    leak: np.ndarray
    received: np.ndarray                          # which acquisitions came back
    t_last_meas_end: int = -1                     # true time the last *syndrome/data* readout ended
    corrections: dict = field(default_factory=dict)   # patch -> CorrectionRecord (final)
    t_correction: dict = field(default_factory=dict)  # patch -> true arrival time
    decisions: dict = field(default_factory=dict)     # register -> (value, true time applied, timed_out)
    t_end: int = -1
    frames_sent: int = 0


class _Sequencer:
    """Executes one ChannelProgram per shot in virtual time."""

    def __init__(self, cs: "ControlSystem", prog: ChannelProgram):
        self.cs = cs
        self.prog = prog
        self.pc = 0
        self.phase = 0
        self.done = True
        self.waiting_on: Optional[str] = None
        self._timeout = None

    def start(self) -> None:
        self.pc = 0
        self.phase = 0
        self.done = False
        self.waiting_on = None
        self._step()

    def _step(self) -> None:
        cs = self.cs
        instrs = self.prog.instrs
        grid = cs.grid_ps
        while True:
            ins = instrs[self.pc]
            op = ins.op
            if op == "WAIT":
                self.pc += 1
                cs.sim.schedule(ins.cycles * grid, self._step)
                return
            if op == "VZ":
                self.phase = (self.phase - ins.quarter) % 4
                self.pc += 1
                continue
            if op in ("PLAY", "ACQUIRE", "RESET"):
                sop = cs.program.ops[ins.op_index]
                if op == "PLAY" and sop.kind in ("SX", "X180") and sop.phase_q != self.phase:
                    raise HardwareFault(f"{self.prog.channel}: frame phase {self.phase} != compiled "
                                        f"{sop.phase_q} at op {sop.index}")
                if op == "RESET":
                    self.phase = 0
                cs._emit(sop)
                self.pc += 1
                cs.sim.schedule(sop.duration_ps, self._step)
                return
            if op == "COND_PLAY":
                sop = cs.program.ops[ins.op_index]
                val = cs.registers.get(ins.register, 0)
                if val == 1:
                    cs._emit(sop)
                cs._decision_applied(ins.register, val)
                self.pc += 1
                cs.sim.schedule(sop.duration_ps, self._step)
                return
            if op == "WAIT_DECISION":
                if ins.register in cs.registers:
                    self.pc += 1
                    continue
                self.waiting_on = ins.register
                cs._waiters.setdefault(ins.register, []).append(self)
                self._timeout = cs.sim.schedule(ins.cycles * grid, self._on_timeout, ins)
                return
            if op == "SYNC":
                raise HardwareFault("SYNC barriers are reserved (not emitted by this compiler version)")
            if op == "MARK":
                cs.metrics.counter(f"mark.{ins.label}").inc()
                self.pc += 1
                continue
            if op == "END":
                self.done = True
                cs._channel_done()
                return
            raise HardwareFault(f"unknown instruction {ins}")

    def wake(self) -> None:
        if self.waiting_on is None:
            return
        self.cs.sim.cancel(self._timeout)
        self.waiting_on = None
        self.pc += 1
        self._step()

    def _on_timeout(self, ins: Instr) -> None:
        cs = self.cs
        if self.waiting_on is None:
            return
        reg = ins.register
        cs.metrics.counter("decision_timeouts").inc()
        cs._set_register(reg, ins.default, timed_out=True)


class ControlSystem:
    def __init__(self, sim: Simulator, qpu: VirtualQPU, timing: Optional[ControlTiming] = None,
                 rng: Optional[RngFactory] = None, metrics: Optional[MetricsRegistry] = None):
        self.sim = sim
        self.qpu = qpu
        self.timing = timing or ControlTiming()
        self.timing.validate()
        rngf = rng or RngFactory(0)
        self.clock = LocalClock(TrueTime(sim), offset_ps=self.timing.clock_offset_ps,
                                freq_ppm=self.timing.clock_ppm, resolution_ps=self.timing.clock_resolution_ps,
                                rng=rngf.stream("control/clock"), name="control")
        self.metrics = metrics or MetricsRegistry("control")
        qpu.readout_sink = self._on_iq
        qpu.sched = sim
        self.link: Optional[SyndromeSink] = None
        self.compiled = None
        self.program: Optional[QpuProgram] = None
        self.grid_ps = 4000
        self._seqs: list[_Sequencer] = []
        self.registers: dict[str, int] = {}
        self._waiters: dict[str, list[_Sequencer]] = {}
        self.records: list[ShotRecord] = []
        self._cur: Optional[ShotRecord] = None
        self._channels_running = 0
        self._shots_remaining = 0
        self._next_shot_id = 0
        self._rep_delay = 0
        self._on_all_done: Optional[Callable[[], None]] = None
        self._pending_bus: list[StaticOp] = []
        self._bus_flush = None
        self._round_state: dict = {}
        self.correction_listeners: list[Callable[[CorrectionRecord], None]] = []
        self.running = False

    # ================================================================ setup
    def attach_link(self, sink: SyndromeSink) -> None:
        self.link = sink

    def load(self, compiled) -> None:
        """Load a CompiledExperiment: upload settings, program, discriminators."""
        if self.running:
            raise HardwareFault("cannot load a program while shots are running")
        self.compiled = compiled
        self.program = compiled.qpu_program
        self.grid_ps = compiled.sequencers.grid_ps
        self.qpu.upload_settings(compiled.settings)
        self.qpu.load_program(compiled.qpu_program)
        self.discriminators: list[Discriminator] = list(compiled.discriminators)
        self._seqs = [_Sequencer(self, ch) for ch in compiled.sequencers.channels]
        man = compiled.manifest
        self._acq_round: dict[int, tuple[int, int]] = {}
        self._round_members: dict[tuple[int, int], list[int]] = {}
        self._final_round: dict[int, int] = {}
        for pm in man.patches:
            self._final_round[pm.patch_id] = pm.rounds
            for r, accs in enumerate(pm.round_acqs):
                self._round_members[(pm.patch_id, r)] = list(accs)
                for a in accs:
                    self._acq_round[a] = (pm.patch_id, r)
        self.metrics.gauge("config_epoch").set(compiled.config_epoch)

    # ================================================================ shot loop
    def run_shots(self, shots: int, rep_delay_ps: Optional[int] = None,
                  on_done: Optional[Callable[[], None]] = None, start_delay_ps: int = 0) -> None:
        if self.program is None:
            raise HardwareFault("no program loaded")
        if self.link is None:
            raise HardwareFault("no AURA-Link transmitter attached")
        if self.running:
            raise HardwareFault("already running")
        self.running = True
        self._shots_remaining = shots
        self._rep_delay = self.compiled.spec.options.rep_delay_ps if rep_delay_ps is None else rep_delay_ps
        self._on_all_done = on_done
        self.sim.schedule(start_delay_ps, self._try_start_shot)

    def _try_start_shot(self) -> None:
        if self._shots_remaining <= 0:
            self.running = False
            if self._on_all_done:
                self._on_all_done()
            return
        if not self.link.can_start_shot():
            self.metrics.counter("shot_start_backpressure").inc()
            self.sim.schedule(self.timing.shot_start_poll_ps, self._try_start_shot)
            return
        sid = self._next_shot_id
        self._next_shot_id += 1
        self._shots_remaining -= 1
        na = self.program.num_acq
        self._cur = ShotRecord(shot=sid, t_start=self.sim.now, bits=np.zeros(na, np.uint8),
                               llr=np.zeros(na), leak=np.zeros(na, bool), received=np.zeros(na, bool))
        self.records.append(self._cur)
        self.registers = {}
        self._waiters = {}
        self._round_state = {}
        self.qpu.begin_shot(sid)
        self._channels_running = len(self._seqs)
        for s in self._seqs:
            s.start()

    def _channel_done(self) -> None:
        self._channels_running -= 1
        if self._channels_running == 0:
            self.qpu.end_shot()
            self._cur.t_end = self.sim.now
            self.metrics.counter("shots").inc()
            self.sim.schedule(self._rep_delay, self._try_start_shot)

    # ================================================================ pulse bus
    def _emit(self, op: StaticOp) -> None:
        self._pending_bus.append(op)
        if self._bus_flush is None:
            self._bus_flush = self.sim.schedule(0, self._flush_bus, priority=10)

    def _flush_bus(self) -> None:
        ops = sorted(self._pending_bus, key=lambda o: o.index)
        self._pending_bus = []
        self._bus_flush = None
        self.qpu.execute(ops)

    # ================================================================ readout path
    def _on_iq(self, shot: int, acq: int, qubit: int, iq: complex) -> None:
        t_meas_end_local = self.clock.timestamp()
        tm = self.timing
        self.sim.schedule(tm.readout_chain_ps + tm.classify_ps, self._classified, shot, acq, qubit, iq,
                          t_meas_end_local, self.sim.now)

    def _classified(self, shot, acq, qubit, iq, ts_local, t_true) -> None:
        bit, llr, leak = self.discriminators[qubit].classify(iq)
        self.sim.schedule(self.timing.crossbar_ps, self._crossbar, shot, acq, int(bit[0]), float(llr[0]),
                          bool(leak[0]), ts_local, t_true)

    def _crossbar(self, shot, acq, bit, llr, leak, ts_local, t_true) -> None:
        rec = self._record(shot)
        if rec is None:
            return
        rec.bits[acq] = bit
        rec.llr[acq] = llr
        rec.leak[acq] = leak
        rec.received[acq] = True
        pr = self._acq_round.get(acq)
        if pr is None:
            return                         # e.g. the branching flag measurement
        rec.t_last_meas_end = max(rec.t_last_meas_end, t_true)
        st = self._round_state.setdefault((shot, pr), {"n": 0, "ts": 0})
        st["n"] += 1
        st["ts"] = max(st["ts"], ts_local)
        if st["n"] == len(self._round_members[pr]):
            self.sim.schedule(self.timing.packetize_ps, self._packetize, shot, pr, st["ts"])

    def _record(self, shot: int) -> Optional[ShotRecord]:
        if self._cur is not None and self._cur.shot == shot:
            return self._cur
        for r in reversed(self.records[-8:]):
            if r.shot == shot:
                return r
        return None

    def _packetize(self, shot: int, pr: tuple[int, int], origin_ts: int) -> None:
        rec = self._record(shot)
        pid, rnd = pr
        accs = self._round_members[pr]
        tm = self.timing
        flags = 0
        if rnd == 0:
            flags |= FLAG_FIRST
        if rnd == self._final_round[pid]:
            flags |= FLAG_LAST
        soft = None
        if tm.soft_bits:
            soft = quantize_llr(rec.llr[accs], tm.soft_bits, tm.llr_step)
            flags |= FLAG_SOFT
        leak = None
        if tm.leak_flags:
            leak = rec.leak[accs].copy()
            flags |= FLAG_LEAK
        srec = SyndromeRecord(patch=pid, shot=shot, round=rnd, bits=rec.bits[accs].copy(), soft=soft,
                              leak=leak, flags=flags, origin_ts=origin_ts)
        rec.frames_sent += 1
        self.metrics.counter("syndrome_frames").inc()
        self.link.send_syndrome(srec)

    # ================================================================ corrections / feed-forward
    def on_correction(self, cor: CorrectionRecord) -> None:
        """Called by the AURA-Link receive path when a COR frame arrives."""
        self.metrics.counter("corrections").inc()
        rec = self._record(cor.shot)
        for fn in self.correction_listeners:
            fn(cor)
        if rec is None:
            self.metrics.counter("corrections_unmatched").inc()
            return
        if cor.round >= self._final_round.get(cor.patch, 1 << 30):
            rec.corrections[cor.patch] = cor
            rec.t_correction[cor.patch] = self.sim.now
            if self._cur is rec and len(cor.obs_flips):
                self.sim.schedule(self.timing.feedback_route_ps, self._set_register, f"L{cor.patch}",
                                  int(cor.obs_flips[0]), False, cor.shot)

    def _set_register(self, reg: str, value: int, timed_out: bool = False, shot: Optional[int] = None) -> None:
        if shot is not None and (self._cur is None or self._cur.shot != shot):
            return
        if reg in self.registers:
            return
        self.registers[reg] = value
        if self._cur is not None:
            self._cur.decisions[reg] = [value, None, timed_out]
        for s in self._waiters.pop(reg, []):
            s.wake()

    def _decision_applied(self, reg: str, value: int) -> None:
        if self._cur is not None and reg in self._cur.decisions:
            self._cur.decisions[reg][1] = self.sim.now

    # ================================================================ results
    def bits_matrix(self) -> np.ndarray:
        return np.array([r.bits for r in self.records], dtype=np.uint8)

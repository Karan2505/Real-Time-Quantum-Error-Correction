"""
AURA-SEQ: the instruction set of the virtual control system's sequencers.

Each hardware channel (one drive/readout channel per qubit, one flux channel
per coupler) runs its own small program, like the per-channel sequencers of
commercial control stacks.  Instructions are timed on the control-system
grid (default 4 ns = one 250 MHz cycle).

Instruction      Semantics
-------------    ---------------------------------------------------------
WAIT n           idle for n grid cycles
VZ q             virtual Z: advance the channel's frame phase by q quarter
                 turns (no pulse; zero time)
PLAY op          emit physical op (SX / X180 / CZ) now; channel busy for its
                 duration.  The pulse axis is the current frame phase.
ACQUIRE op       start a measurement (MEASURE op) now; busy for its duration
RESET op         unconditional active reset now
SYNC id          barrier: wait until every channel listed for ``id`` arrives
WAIT_DECISION r  block until decision register r is written by the feedback
                 unit, or until the timeout (then r := default)
COND_PLAY r op   emit op only if decision register r == 1
                 (the op's duration is consumed either way: deterministic timing)
MARK label       timestamp marker for telemetry (zero time)
END              end of the shot for this channel
"""

from __future__ import annotations

from dataclasses import dataclass, field

OPCODES = ("WAIT", "VZ", "PLAY", "ACQUIRE", "RESET", "SYNC", "WAIT_DECISION", "COND_PLAY", "MARK", "END")


@dataclass(frozen=True)
class Instr:
    op: str
    cycles: int = 0            # WAIT length / WAIT_DECISION timeout
    op_index: int = -1         # static QPU op for PLAY / ACQUIRE / RESET / COND_PLAY
    quarter: int = 0           # VZ amount; PLAY expected frame phase (self-check)
    register: str = ""         # WAIT_DECISION / COND_PLAY
    barrier: str = ""          # SYNC id
    label: str = ""            # MARK
    default: int = 0           # WAIT_DECISION value on timeout

    def __post_init__(self) -> None:
        if self.op not in OPCODES:
            raise ValueError(f"unknown opcode {self.op}")
        if self.op == "WAIT" and self.cycles <= 0:
            raise ValueError("WAIT needs a positive cycle count")
        if self.op in ("PLAY", "ACQUIRE", "RESET", "COND_PLAY") and self.op_index < 0:
            raise ValueError(f"{self.op} needs an op index")

    def __str__(self) -> str:  # assembler-style listing
        if self.op == "WAIT":
            return f"WAIT {self.cycles}"
        if self.op == "VZ":
            return f"VZ {self.quarter}"
        if self.op in ("PLAY", "ACQUIRE", "RESET"):
            return f"{self.op} #{self.op_index}"
        if self.op == "COND_PLAY":
            return f"COND_PLAY {self.register} #{self.op_index}"
        if self.op == "WAIT_DECISION":
            return f"WAIT_DECISION {self.register} timeout={self.cycles}"
        if self.op == "SYNC":
            return f"SYNC {self.barrier}"
        if self.op == "MARK":
            return f"MARK {self.label}"
        return self.op


@dataclass(frozen=True)
class ChannelProgram:
    channel: str               # 'q<idx>' or 'c<idx>'
    kind: str                  # 'qubit' | 'coupler'
    target: int                # qubit or coupler index
    instrs: tuple[Instr, ...]

    def listing(self) -> str:
        return "\n".join(f"{i:5d}  {ins}" for i, ins in enumerate(self.instrs))


@dataclass(frozen=True)
class SequencerBundle:
    """All channel programs of one compiled experiment + barrier membership."""
    channels: tuple[ChannelProgram, ...]
    barriers: dict = field(default_factory=dict)      # barrier id -> tuple of channel names
    grid_ps: int = 4000

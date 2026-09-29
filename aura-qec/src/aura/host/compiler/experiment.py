"""Experiment specifications accepted by the QEC compiler."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ...codes.css import CSSCode
from ...core.errors import CompileError


@dataclass(frozen=True)
class PatchSpec:
    """One logical block (code patch) in the experiment."""
    patch_id: int
    code: CSSCode
    basis: str = "Z"          # memory basis: 'Z' (prepare/measure |0_L>) or 'X' (|+_L>)
    rounds: int = 3

    def validate(self) -> None:
        if self.basis not in ("Z", "X"):
            raise CompileError(f"patch {self.patch_id}: basis must be 'Z' or 'X'")
        if self.rounds < 1:
            raise CompileError(f"patch {self.patch_id}: need at least one round")
        if self.basis == "X" and self.code.num_x_checks == 0:
            raise CompileError(f"patch {self.patch_id}: code {self.code.name} has no X checks; "
                               "use basis 'Z'")
        if self.basis == "Z" and self.code.num_z_checks == 0:
            raise CompileError(f"patch {self.patch_id}: code {self.code.name} has no Z checks")
        if not 0 <= self.patch_id < 65536:
            raise CompileError("patch id must fit in 16 bits")


@dataclass(frozen=True)
class CompileOptions:
    schedule: str = "auto"            # 'auto' | 'interleaved' | 'separated'
    dynamical_decoupling: bool = True
    dd_min_idle_ps: int = 200_000     # insert DD in idle windows >= 200 ns on data qubits
    measurement_barrier: bool = True  # all ancillas of a round start readout together
    build_dem: bool = True
    allow_uncalibrated: bool = False
    branching: bool = False           # append a decoder-conditioned X on a flag qubit (feed-forward demo)
    decision_timeout_ps: int = 50_000_000   # 50 us
    rep_delay_ps: int = 2_000_000     # idle time between shots (2 us)


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    patches: tuple[PatchSpec, ...]
    options: CompileOptions = field(default_factory=CompileOptions)

    def validate(self) -> None:
        if not self.patches:
            raise CompileError("experiment has no patches")
        ids = [p.patch_id for p in self.patches]
        if len(ids) != len(set(ids)):
            raise CompileError("duplicate patch ids")
        for p in self.patches:
            p.validate()
        if self.options.schedule not in ("auto", "interleaved", "separated"):
            raise CompileError(f"unknown schedule mode {self.options.schedule}")


def memory_experiment(code: CSSCode, rounds: int, basis: str = "Z", patch_id: int = 0,
                      name: Optional[str] = None, **opts) -> ExperimentSpec:
    return ExperimentSpec(name=name or f"memory-{code.name}-{basis}-r{rounds}",
                          patches=(PatchSpec(patch_id, code, basis, rounds),),
                          options=CompileOptions(**opts))

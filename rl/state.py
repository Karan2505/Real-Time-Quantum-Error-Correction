"""
rl/state.py
===========
Typed RL state representation and its conversion to a numeric vector.

Feature ordering (ORACLE mode, 5 dimensions) — this order is fixed and
must not change without also updating every consumer (environment,
DQN input layer, saved checkpoints):

    index 0: unsatisfied_check_ratio
    index 1: delta_LLR
    index 2: current_iteration
    index 3: physical_error_rate      <- oracle-only, see StateMode below
    index 4: iterations_since_improvement

REALISTIC mode (4 dimensions) drops index 3 entirely rather than
zeroing it, so the two modes are never numerically confusable and a
REALISTIC-trained network cannot accidentally be fed a 5-wide vector:

    index 0: unsatisfied_check_ratio
    index 1: delta_LLR
    index 2: current_iteration
    index 3: iterations_since_improvement

Why the distinction matters
----------------------------
``physical_error_rate`` is a simulation parameter: in Stim-style
training it is known exactly because *we* generated the noise. A
real deployed decoder generally does not have direct, trustworthy
access to the true physical error rate of the hardware at decode
time — at best it might have a slow, indirectly-estimated proxy.
Training and evaluating in ORACLE mode is fine for infrastructure
development and upper-bound studies, but any claim about realistic
deployability must use REALISTIC mode, and the two must never be
mixed within one comparison (see project spec, "Avoid Data Leakage").
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class StateMode(str, Enum):
    """Which feature set a :class:`DecoderState` is converted into."""

    ORACLE = "oracle"
    REALISTIC = "realistic"


ORACLE_FEATURE_NAMES = (
    "unsatisfied_check_ratio",
    "delta_LLR",
    "current_iteration",
    "physical_error_rate",
    "iterations_since_improvement",
)

REALISTIC_FEATURE_NAMES = (
    "unsatisfied_check_ratio",
    "delta_LLR",
    "current_iteration",
    "iterations_since_improvement",
)


@dataclass(frozen=True)
class DecoderState:
    """A single decoder-state observation.

    ``physical_error_rate`` is always stored (it is convenient to keep
    around for logging/analysis even in realistic-mode experiments) but
    is only included in the numeric vector when ``mode=StateMode.ORACLE``.
    """

    unsatisfied_check_ratio: float
    delta_LLR: float
    current_iteration: int
    physical_error_rate: float
    iterations_since_improvement: int

    def __post_init__(self) -> None:
        if not (0.0 <= self.unsatisfied_check_ratio <= 1.0):
            raise ValueError(
                "unsatisfied_check_ratio must be in [0, 1], got "
                f"{self.unsatisfied_check_ratio}."
            )
        if self.current_iteration < 0:
            raise ValueError("current_iteration must be >= 0.")
        if not (0.0 <= self.physical_error_rate <= 1.0):
            raise ValueError(
                "physical_error_rate must be in [0, 1], got "
                f"{self.physical_error_rate}."
            )
        if self.iterations_since_improvement < 0:
            raise ValueError("iterations_since_improvement must be >= 0.")

    def to_vector(self, mode: StateMode = StateMode.ORACLE) -> np.ndarray:
        """Convert to a deterministic-order ``np.float32`` vector.

        Parameters
        ----------
        mode:
            ``StateMode.ORACLE`` includes ``physical_error_rate`` (5-dim,
            matching the project spec's DQN input size). ``StateMode.
            REALISTIC`` excludes it (4-dim), representing what a
            deployed decoder could plausibly observe.
        """
        if mode == StateMode.ORACLE:
            return np.array(
                [
                    self.unsatisfied_check_ratio,
                    self.delta_LLR,
                    float(self.current_iteration),
                    self.physical_error_rate,
                    float(self.iterations_since_improvement),
                ],
                dtype=np.float32,
            )
        elif mode == StateMode.REALISTIC:
            return np.array(
                [
                    self.unsatisfied_check_ratio,
                    self.delta_LLR,
                    float(self.current_iteration),
                    float(self.iterations_since_improvement),
                ],
                dtype=np.float32,
            )
        raise ValueError(f"Unknown StateMode: {mode!r}")

    @staticmethod
    def feature_names(mode: StateMode = StateMode.ORACLE) -> tuple:
        return ORACLE_FEATURE_NAMES if mode == StateMode.ORACLE else REALISTIC_FEATURE_NAMES

    @staticmethod
    def vector_size(mode: StateMode = StateMode.ORACLE) -> int:
        return len(DecoderState.feature_names(mode))

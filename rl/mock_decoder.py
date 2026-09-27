"""
rl/mock_decoder.py
===================
MOCK / PLACEHOLDER iterative decoder.

================================================================
 THIS IS A SYNTHETIC/MOCK IMPLEMENTATION FOR RL INFRASTRUCTURE
 TESTING. IT IS NOT A QUANTUM ERROR-CORRECTION DECODER. IT DOES
 NOT IMPLEMENT RELAY-BP, BELIEF PROPAGATION, OR ANY QLDPC/STIM
 MODEL, AND ITS OUTPUT MUST NEVER BE REPORTED AS A QEC RESULT.
================================================================

Purpose
-------
The RL environment (rl/environment.py) needs *something* that behaves
like an iterative decoder — i.e. something with a notion of "current
iteration", a quantity that trends toward convergence, and occasional
stagnation — so the RL state/action/reward/environment machinery can be
built and unit tested before a real Relay-BP implementation exists
anywhere in this project (see Phase 0 report: no Relay-BP, OSD, Stim,
or qLDPC implementation currently exists in this repository or in
Karan's Deltaflow validation framework).

This class produces a synthetic ``unsatisfied_check_ratio`` trajectory
using a simple deterministic-given-seed geometric decay plus Gaussian
noise. It has no relationship to real BP/LLR dynamics beyond sharing
the same *shape* of problem (a ratio that should trend down, with
occasional stalls) needed to exercise CONTINUE/STOP/OSD logic.

Determinism
-----------
All randomness is drawn from a ``numpy.random.Generator`` seeded in
``reset()``. Two decoders reset with the same
``(physical_error_rate, seed)`` produce bit-identical trajectories.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rl.config import MockDecoderConfig
from rl.state import DecoderState


@dataclass
class MockIterationResult:
    """Return value of a single mock CONTINUE/OSD step."""

    state: DecoderState
    improved: bool


class MockRelayBPDecoder:
    """MOCK / PLACEHOLDER stand-in for an iterative Relay-BP decoder.

    See module docstring: this is not a real decoder. It exists only so
    rl/environment.py has something to drive during Phase-1 development.
    """

    def __init__(self, config: MockDecoderConfig | None = None) -> None:
        self._config = config or MockDecoderConfig()
        self._rng: np.random.Generator | None = None
        self._physical_error_rate: float = 0.0
        self._iteration: int = 0
        self._unsatisfied_ratio: float = 0.0
        self._prev_unsatisfied_ratio: float = 0.0
        self._iterations_since_improvement: int = 0
        self._initialized = False

    @property
    def config(self) -> MockDecoderConfig:
        return self._config

    @property
    def current_iteration(self) -> int:
        return self._iteration

    def reset(self, physical_error_rate: float, seed: int) -> DecoderState:
        """Start a new synthetic decoding episode.

        Parameters
        ----------
        physical_error_rate:
            Simulation-side noise parameter in [0, 1]. Used only to pick
            a synthetic starting ``unsatisfied_check_ratio`` — this is
            not a physically calibrated relationship.
        seed:
            Seeds this decoder's internal RNG. Same
            ``(physical_error_rate, seed)`` -> identical trajectory.
        """
        if not (0.0 <= physical_error_rate <= 1.0):
            raise ValueError("physical_error_rate must be in [0, 1].")

        self._rng = np.random.default_rng(seed)
        self._physical_error_rate = physical_error_rate
        self._iteration = 0
        self._iterations_since_improvement = 0

        initial = float(
            np.clip(
                physical_error_rate * self._config.initial_ratio_scale
                + self._rng.normal(0.0, self._config.noise_scale),
                0.0,
                1.0,
            )
        )
        self._unsatisfied_ratio = initial
        self._prev_unsatisfied_ratio = initial
        self._initialized = True

        return self._current_state(delta_LLR=0.0)

    def step_continue(self) -> MockIterationResult:
        """Advance the synthetic decoder by one CONTINUE iteration."""
        self._require_initialized()

        self._prev_unsatisfied_ratio = self._unsatisfied_ratio
        noise = self._rng.normal(0.0, self._config.noise_scale)
        new_ratio = float(
            np.clip(
                self._unsatisfied_ratio * self._config.decay_factor + noise,
                0.0,
                1.0,
            )
        )
        self._unsatisfied_ratio = new_ratio
        self._iteration += 1

        delta = self._prev_unsatisfied_ratio - new_ratio  # positive = improvement
        improved = delta >= self._config.improvement_threshold
        if improved:
            self._iterations_since_improvement = 0
        else:
            self._iterations_since_improvement += 1

        return MockIterationResult(
            state=self._current_state(delta_LLR=delta),
            improved=improved,
        )

    def is_converged(self) -> bool:
        """Whether the current synthetic ratio counts as a successful stop."""
        self._require_initialized()
        return self._unsatisfied_ratio <= self._config.convergence_threshold

    def osd_fallback(self) -> bool:
        """MOCK OSD / PLACEHOLDER.

        Deterministically (given the seeded RNG stream) decides whether
        the synthetic OSD fallback "succeeds", biased toward success
        since real OSD post-processing is a stronger, slower solver
        typically invoked precisely because plain BP has stalled. This
        has no relationship to a real OSD implementation's behavior.
        """
        self._require_initialized()
        # Higher unsatisfied_ratio -> lower (but still improved-over-BP)
        # synthetic OSD success probability. Purely illustrative.
        success_probability = float(np.clip(1.0 - 0.5 * self._unsatisfied_ratio, 0.05, 0.99))
        return bool(self._rng.random() < success_probability)

    def _current_state(self, delta_LLR: float) -> DecoderState:
        return DecoderState(
            unsatisfied_check_ratio=self._unsatisfied_ratio,
            delta_LLR=delta_LLR,
            current_iteration=self._iteration,
            physical_error_rate=self._physical_error_rate,
            iterations_since_improvement=self._iterations_since_improvement,
        )

    def _require_initialized(self) -> None:
        if not self._initialized:
            raise RuntimeError(
                "MockRelayBPDecoder.reset() must be called before "
                "step_continue()/is_converged()/osd_fallback()."
            )

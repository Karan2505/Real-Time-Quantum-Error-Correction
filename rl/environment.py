"""
rl/environment.py
==================
Decoding-episode environment for the RL controller.

Gymnasium dependency decision
------------------------------
Gymnasium is NOT added as a dependency in Phase 1. Rationale:

  * The existing repository's dependency footprint is intentionally
    minimal (pytest, pytest-timeout, numpy, pyyaml, rich — see
    requirements.txt); Phase 1 only needs a `reset()`/`step()` loop
    over a single, non-vectorized, discrete-action environment, which
    is a small amount of code to hand-write.
  * Phase 1's decoder is a synthetic mock (see rl/mock_decoder.py) with
    no rendering, no need for Gymnasium's registry/vector-env/wrapper
    ecosystem, and no external users needing Gymnasium-compatible
    tooling (SB3, RLlib, etc.) yet.
  * The public API below intentionally mirrors Gymnasium's `reset()` /
    `step()` contract (`observation, info` / `observation, reward,
    terminated, truncated, info`) so that adopting real Gymnasium later
    — e.g. once the real Relay-BP decoder exists and we want SB3/RLlib
    baselines — is a near-drop-in change rather than a rewrite.

If a future phase needs vectorized environments, standard wrappers, or
integration with an existing RL library that expects a real
`gymnasium.Env` subclass, adding the dependency then is the right call;
revisit this decision at that point rather than defaulting into it now.

Decoupling from the decoder implementation
-------------------------------------------
This environment depends only on ``DecoderProtocol`` below, not on any
concrete decoder class. Phase 1 wires it to
:class:`rl.mock_decoder.MockRelayBPDecoder`. When a real Relay-BP
implementation exists, an adapter satisfying the same protocol can be
substituted with no change to this file. This environment also does
NOT import from or depend on ``harness/qeci_interface.py`` — that
remains Karan's chunk-level transport abstraction and is out of scope
for Phase 1 (see project spec, "Integration requirement").
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np

from rl.actions import Action, coerce_action
from rl.config import EnvironmentConfig, RewardConfig
from rl.reward import continue_step_reward, terminal_reward
from rl.state import DecoderState, StateMode


@runtime_checkable
class DecoderProtocol(Protocol):
    """Minimal interface the environment needs from *any* decoder
    backend (mock today, real Relay-BP later).

    Concrete implementations must be deterministic given
    ``(physical_error_rate, seed)`` passed to ``reset``.
    """

    def reset(self, physical_error_rate: float, seed: int) -> DecoderState:
        ...

    def step_continue(self) -> Any:
        """Advance one iteration. Must return an object with a
        ``.state: DecoderState`` attribute (see
        :class:`rl.mock_decoder.MockIterationResult`)."""
        ...

    def is_converged(self) -> bool:
        ...

    def osd_fallback(self) -> bool:
        """Resolve an OSD (ACTION 2) call and report whether the
        resulting correction is judged successful.

        Renamed from ``mock_osd_fallback`` (Phase 1) now that a real
        backend exists: Phase 1's :class:`rl.mock_decoder.
        MockRelayBPDecoder` implements this with a labeled synthetic
        stand-in; Phase 2's :class:`rl.bp_decoder_adapter.
        BpDecoderAdapter` implements it with real OSD
        (``ldpc.BpOsdDecoder``)."""
        ...


@dataclass
class StepInfo:
    """Structured extra information returned in ``step()``'s info dict.

    Kept as a dataclass (and also expanded into a plain dict, since
    Gym-style consumers expect ``info`` to be a dict) so field names are
    documented in one place instead of scattered string literals.
    """

    action_taken: Action
    wcet_guard_activated: bool
    used_osd: bool
    success: bool | None
    deadline_violated: bool
    current_iteration: int

    def as_dict(self) -> dict:
        return {
            "action_taken": int(self.action_taken),
            "wcet_guard_activated": self.wcet_guard_activated,
            "used_osd": self.used_osd,
            "success": self.success,
            "deadline_violated": self.deadline_violated,
            "current_iteration": self.current_iteration,
        }


class QECDecodingEnv:
    """Single-episode decoding environment with a Gymnasium-shaped API.

    NOTE: Backed by a MOCK decoder in Phase 1 (see rl/mock_decoder.py).
    Metrics, rewards, and "success" produced by this environment are
    infrastructure-testing artifacts, not QEC research results.
    """

    def __init__(
        self,
        decoder: DecoderProtocol,
        env_config: EnvironmentConfig | None = None,
        reward_config: RewardConfig | None = None,
    ) -> None:
        self._decoder = decoder
        self._env_config = env_config or EnvironmentConfig()
        self._reward_config = reward_config or RewardConfig()
        self._state_mode = StateMode(self._env_config.state_mode)

        self._current_state: DecoderState | None = None
        self._episode_active = False

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def state_mode(self) -> StateMode:
        return self._state_mode

    @property
    def observation_size(self) -> int:
        return DecoderState.vector_size(self._state_mode)

    @property
    def env_config(self) -> EnvironmentConfig:
        return self._env_config

    @property
    def reward_config(self) -> RewardConfig:
        return self._reward_config

    # ------------------------------------------------------------------
    # Gymnasium-shaped API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int = 0,
        physical_error_rate: float | None = None,
    ) -> tuple[np.ndarray, dict]:
        """Start a new episode.

        Returns
        -------
        (observation, info) — matching Gymnasium's ``reset()`` contract.
        """
        per = (
            physical_error_rate
            if physical_error_rate is not None
            else self._env_config.default_physical_error_rate
        )
        self._current_state = self._decoder.reset(physical_error_rate=per, seed=seed)
        self._episode_active = True

        obs = self._current_state.to_vector(self._state_mode)
        info = StepInfo(
            action_taken=Action.CONTINUE,  # no action taken yet; placeholder
            wcet_guard_activated=False,
            used_osd=False,
            success=None,
            deadline_violated=False,
            current_iteration=self._current_state.current_iteration,
        ).as_dict()
        return obs, info

    def step(
        self, action: int | Action
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Advance the episode by one decision.

        Returns
        -------
        (observation, reward, terminated, truncated, info) — matching
        Gymnasium's ``step()`` contract.

        Raises
        ------
        RuntimeError
            If called before ``reset()`` or after the episode ended.
        ValueError
            If ``action`` is not a valid :class:`Action`.
        """
        if not self._episode_active or self._current_state is None:
            raise RuntimeError("step() called before reset() or after episode end.")

        requested_action = coerce_action(action)
        cfg = self._env_config

        # --- WCET guard -------------------------------------------------
        # Deterministic override: if CONTINUE was requested but we are at
        # or past the configured deadline, force a safe STOP instead.
        # This is a configurable abstraction, not a proven real-time
        # guarantee (see EnvironmentConfig.wcet_deadline_iterations).
        wcet_guard_activated = False
        effective_action = requested_action
        if (
            requested_action == Action.CONTINUE
            and self._current_state.current_iteration >= cfg.wcet_deadline_iterations
        ):
            effective_action = Action.STOP
            wcet_guard_activated = True

        if effective_action == Action.CONTINUE:
            return self._do_continue(requested_action, wcet_guard_activated)
        elif effective_action == Action.STOP:
            return self._do_stop(requested_action, wcet_guard_activated)
        elif effective_action == Action.OSD:
            return self._do_osd(requested_action, wcet_guard_activated)
        raise AssertionError(f"Unhandled action: {effective_action!r}")  # pragma: no cover

    # ------------------------------------------------------------------
    # Action handlers
    # ------------------------------------------------------------------

    def _do_continue(
        self, requested_action: Action, wcet_guard_activated: bool
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        result = self._decoder.step_continue()
        self._current_state = result.state

        reward = continue_step_reward(self._reward_config)
        terminated = False
        truncated = False
        deadline_violated = False

        # Defensive hard cap: the WCET guard should always intervene
        # before this point is reached (guard threshold <=
        # max_iterations), but if it is ever misconfigured or reached
        # anyway, do not allow a silent unterminated episode — end it
        # and flag a genuine deadline violation.
        if self._current_state.current_iteration >= self._env_config.max_iterations:
            truncated = True
            deadline_violated = True
            reward += terminal_reward(
                success=False,
                used_osd=False,
                deadline_violated=True,
                cfg=self._reward_config,
            )
            self._episode_active = False

        obs = self._current_state.to_vector(self._state_mode)
        info = StepInfo(
            action_taken=requested_action,
            wcet_guard_activated=wcet_guard_activated,
            used_osd=False,
            success=(False if deadline_violated else None),
            deadline_violated=deadline_violated,
            current_iteration=self._current_state.current_iteration,
        ).as_dict()
        return obs, reward, terminated, truncated, info

    def _do_stop(
        self, requested_action: Action, wcet_guard_activated: bool
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        success = self._decoder.is_converged()
        reward = terminal_reward(
            success=success,
            used_osd=False,
            deadline_violated=False,
            cfg=self._reward_config,
        )
        self._episode_active = False

        obs = self._current_state.to_vector(self._state_mode)
        info = StepInfo(
            action_taken=requested_action,
            wcet_guard_activated=wcet_guard_activated,
            used_osd=False,
            success=success,
            deadline_violated=False,
            current_iteration=self._current_state.current_iteration,
        ).as_dict()
        return obs, reward, True, False, info

    def _do_osd(
        self, requested_action: Action, wcet_guard_activated: bool
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        # MOCK OSD / PLACEHOLDER — see rl/mock_decoder.py::osd_fallback
        success = self._decoder.osd_fallback()
        reward = terminal_reward(
            success=success,
            used_osd=True,
            deadline_violated=False,
            cfg=self._reward_config,
        )
        self._episode_active = False

        obs = self._current_state.to_vector(self._state_mode)
        info = StepInfo(
            action_taken=requested_action,
            wcet_guard_activated=wcet_guard_activated,
            used_osd=True,
            success=success,
            deadline_violated=False,
            current_iteration=self._current_state.current_iteration,
        ).as_dict()
        return obs, reward, True, False, info

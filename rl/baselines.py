"""
rl/baselines.py
================
Phase 3: deterministic baseline policies, and the evaluation harness
used to compare them (and, later, a trained RL policy) against each
other on the same environment/decoder.

These baselines exist so that any future RL result has something
honest to be measured against — per the project spec: "Before claiming
that RL improves anything, implement or expose baseline policies."

Policies here are plain Python objects with a ``select_action(obs,
info) -> Action`` method — no learning, no state beyond what's passed
in each call (aside from configuration set at construction time). They
work against ANY decoder satisfying ``rl.environment.DecoderProtocol``
— the mock (Phase 1) or the real BP/OSD adapter (Phase 2) — since they
only look at the observation vector and info dict, never the decoder
directly.

Observation indexing note
--------------------------
``obs[2]`` is ``current_iteration`` and ``obs[-1]`` is
``iterations_since_improvement`` in BOTH oracle mode (5-dim) and
realistic mode (4-dim) — see rl/state.py's ORACLE_FEATURE_NAMES /
REALISTIC_FEATURE_NAMES, which deliberately share this prefix/suffix
ordering. Baselines below rely on this and work under either mode.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from typing import Protocol, Sequence

import numpy as np

from rl.actions import Action
from rl.environment import QECDecodingEnv


class BaselinePolicy(Protocol):
    """Minimal interface every baseline (and, eventually, a trained RL
    policy wrapper) must satisfy to be run by ``run_episode``/
    ``evaluate_policy`` below."""

    name: str

    def select_action(self, obs: np.ndarray, info: dict) -> Action:
        ...


@dataclass
class AlwaysContinuePolicy:
    """Baseline A: keep running Relay-BP/BP until something else (the
    WCET guard, or the environment's own hard iteration cap) stops it.
    Never chooses STOP or OSD itself."""

    name: str = "always_continue"

    def select_action(self, obs: np.ndarray, info: dict) -> Action:
        return Action.CONTINUE


@dataclass
class FixedIterationStopPolicy:
    """Baseline B: run CONTINUE until a predetermined iteration count,
    then STOP — regardless of whether the decoder has actually
    converged. Never uses OSD.

    Attributes
    ----------
    stop_at_iteration:
        STOP is chosen once ``current_iteration >= stop_at_iteration``.
    """

    stop_at_iteration: int
    name: str = "fixed_iteration_stop"

    def __post_init__(self) -> None:
        if self.stop_at_iteration <= 0:
            raise ValueError("stop_at_iteration must be > 0.")

    def select_action(self, obs: np.ndarray, info: dict) -> Action:
        current_iteration = int(obs[2])
        return Action.STOP if current_iteration >= self.stop_at_iteration else Action.CONTINUE


@dataclass
class DeterministicStagnationFallbackPolicy:
    """Baseline C: run CONTINUE normally; once the decoder has gone
    ``stagnation_threshold`` iterations without meaningful improvement,
    invoke OSD instead. This is the "if stagnation: OSD" rule from the
    project spec — deterministic, not learned.

    Attributes
    ----------
    stagnation_threshold:
        OSD is chosen once
        ``iterations_since_improvement >= stagnation_threshold``.
    """

    stagnation_threshold: int
    name: str = "deterministic_stagnation_fallback"

    def __post_init__(self) -> None:
        if self.stagnation_threshold <= 0:
            raise ValueError("stagnation_threshold must be > 0.")

    def select_action(self, obs: np.ndarray, info: dict) -> Action:
        iterations_since_improvement = int(obs[-1])
        if iterations_since_improvement >= self.stagnation_threshold:
            return Action.OSD
        return Action.CONTINUE


@dataclass
class EpisodeResult:
    """Outcome of running one policy through one episode."""

    policy_name: str
    seed: int
    physical_error_rate: float
    decision_steps: int  # number of env.step() calls (CONTINUE/STOP/OSD decisions)
    final_iteration: int  # decoder iteration count at episode end
    total_reward: float
    success: bool
    used_osd: bool
    deadline_violated: bool
    wcet_guard_activations: int


def run_episode(
    env: QECDecodingEnv,
    policy: BaselinePolicy,
    seed: int,
    physical_error_rate: float,
) -> EpisodeResult:
    """Run exactly one episode of ``env`` under ``policy`` and return
    its outcome. Works against any decoder backend behind ``env``."""
    obs, info = env.reset(seed=seed, physical_error_rate=physical_error_rate)
    total_reward = 0.0
    wcet_guard_activations = 0
    decision_steps = 0
    terminated = truncated = False

    while not (terminated or truncated):
        action = policy.select_action(obs, info)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        decision_steps += 1
        if info["wcet_guard_activated"]:
            wcet_guard_activations += 1

    return EpisodeResult(
        policy_name=policy.name,
        seed=seed,
        physical_error_rate=physical_error_rate,
        decision_steps=decision_steps,
        final_iteration=info["current_iteration"],
        total_reward=total_reward,
        success=bool(info["success"]),
        used_osd=bool(info["used_osd"]),
        deadline_violated=bool(info["deadline_violated"]),
        wcet_guard_activations=wcet_guard_activations,
    )


@dataclass
class AggregateResult:
    """Summary statistics over multiple episodes/seeds for one policy
    at one physical error rate — the per-policy, per-noise-level row
    the project spec's experiment sweeps (Section 16) ultimately need.
    """

    policy_name: str
    physical_error_rate: float
    n_episodes: int
    success_rate: float
    mean_final_iteration: float
    median_final_iteration: float
    osd_rate: float
    deadline_violation_rate: float
    mean_wcet_guard_activations: float
    mean_reward: float


def evaluate_policy(
    env: QECDecodingEnv,
    policy: BaselinePolicy,
    seeds: Sequence[int],
    physical_error_rate: float,
) -> AggregateResult:
    """Run ``policy`` over every seed in ``seeds`` at a fixed
    ``physical_error_rate`` and aggregate the results.

    Per project spec ("Use multiple random seeds. Report mean and
    variability... Do not use a single random train/test split and
    claim generalization."), this always runs multiple episodes rather
    than reporting a single run.
    """
    if not seeds:
        raise ValueError("seeds must be non-empty.")

    results = [run_episode(env, policy, seed, physical_error_rate) for seed in seeds]
    n = len(results)
    final_iterations = [r.final_iteration for r in results]

    return AggregateResult(
        policy_name=policy.name,
        physical_error_rate=physical_error_rate,
        n_episodes=n,
        success_rate=sum(r.success for r in results) / n,
        mean_final_iteration=float(np.mean(final_iterations)),
        median_final_iteration=float(median(final_iterations)),
        osd_rate=sum(r.used_osd for r in results) / n,
        deadline_violation_rate=sum(r.deadline_violated for r in results) / n,
        mean_wcet_guard_activations=float(
            np.mean([r.wcet_guard_activations for r in results])
        ),
        mean_reward=float(np.mean([r.total_reward for r in results])),
    )


def compare_policies(
    env: QECDecodingEnv,
    policies: Sequence[BaselinePolicy],
    seeds: Sequence[int],
    physical_error_rate: float,
) -> list[AggregateResult]:
    """Convenience wrapper: evaluate several policies under identical
    conditions (same env/decoder, same seeds, same noise level) so
    their ``AggregateResult``s are directly comparable."""
    return [evaluate_policy(env, policy, seeds, physical_error_rate) for policy in policies]

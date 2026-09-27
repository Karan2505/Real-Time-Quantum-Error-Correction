"""
rl/reward.py
============
Reward computation, kept separate from environment.py so the reward
formula can be unit-tested (and later swept for sensitivity analysis,
per project spec Section 13/16) independently of episode mechanics.

Design
------
reward = terminal_component - iteration_cost_component

Terminal component (paid once, when the episode ends via STOP or OSD,
or via the WCET guard's forced stop):

    + correct_reward   if the accepted correction is judged successful
    + wrong_penalty     otherwise (wrong_penalty is already negative)
    - osd_cost          additionally, if OSD was used
    + deadline_penalty  additionally (already negative), if the episode
                        ended via a true deadline violation rather than
                        a normal STOP/OSD/guard-forced-stop

Per-iteration component (paid on every CONTINUE step, not just at the
end): a fixed ``iteration_cost`` per step, so two runs that both
succeed but took different numbers of iterations are still ranked by
efficiency (project spec: "Correct but unnecessarily slow -> smaller
positive reward").

This module intentionally does not "invent" any additional scientific
justification for the numbers themselves; see rl/config.py for that
caveat.
"""

from __future__ import annotations

from rl.config import RewardConfig


def continue_step_reward(cfg: RewardConfig) -> float:
    """Reward (always <= 0) charged for a single CONTINUE iteration."""
    return -cfg.iteration_cost


def terminal_reward(
    *,
    success: bool,
    used_osd: bool,
    deadline_violated: bool,
    cfg: RewardConfig,
) -> float:
    """Reward charged once when an episode ends.

    Parameters
    ----------
    success:
        Whether the final accepted correction is judged correct
        (synthetic convergence in Phase 1; a real oracle/logical-error
        check once a real decoder exists).
    used_osd:
        Whether OSD (ACTION 2) was the terminating action.
    deadline_violated:
        Whether the episode ended because the hard iteration budget
        (``EnvironmentConfig.max_iterations``) was exhausted without a
        STOP/OSD decision — i.e. the WCET guard's own final safety net
        had to trigger. This should be rare/impossible by construction
        (the guard forces STOP before reaching that point) and mainly
        exists as a defensive check; see rl/environment.py.
    cfg:
        Reward coefficients.
    """
    reward = cfg.correct_reward if success else cfg.wrong_penalty
    if used_osd:
        reward -= cfg.osd_cost
    if deadline_violated:
        reward += cfg.deadline_penalty  # already <= 0
    return reward

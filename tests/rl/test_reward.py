"""Unit tests for rl/reward.py.

Per project spec: don't just pin exact arbitrary numbers — also assert
the intended qualitative ordering/behavior (correct+fast > correct+slow
> wrong; unnecessary OSD is penalized; deadline violation is worst).
"""

from __future__ import annotations

import pytest

from rl.config import RewardConfig
from rl.reward import continue_step_reward, terminal_reward


@pytest.fixture
def cfg() -> RewardConfig:
    return RewardConfig(
        correct_reward=10.0,
        wrong_penalty=-10.0,
        iteration_cost=0.1,
        osd_cost=2.0,
        deadline_penalty=-50.0,
    )


class TestContinueStepReward:
    def test_is_negative(self, cfg):
        assert continue_step_reward(cfg) < 0.0

    def test_matches_configured_cost(self, cfg):
        assert continue_step_reward(cfg) == -cfg.iteration_cost


class TestTerminalReward:
    def test_success_gives_positive_reward(self, cfg):
        r = terminal_reward(success=True, used_osd=False, deadline_violated=False, cfg=cfg)
        assert r == cfg.correct_reward
        assert r > 0

    def test_failure_gives_negative_reward(self, cfg):
        r = terminal_reward(success=False, used_osd=False, deadline_violated=False, cfg=cfg)
        assert r == cfg.wrong_penalty
        assert r < 0

    def test_success_strictly_beats_failure(self, cfg):
        success_r = terminal_reward(success=True, used_osd=False, deadline_violated=False, cfg=cfg)
        failure_r = terminal_reward(success=False, used_osd=False, deadline_violated=False, cfg=cfg)
        assert success_r > failure_r

    def test_unnecessary_osd_has_a_cost(self, cfg):
        # Same outcome (success), but OSD was used: strictly worse than
        # a plain-BP success, because osd_cost > 0.
        plain_success = terminal_reward(success=True, used_osd=False, deadline_violated=False, cfg=cfg)
        osd_success = terminal_reward(success=True, used_osd=True, deadline_violated=False, cfg=cfg)
        assert osd_success < plain_success
        assert osd_success == pytest.approx(plain_success - cfg.osd_cost)

    def test_osd_failure_also_penalized_relative_to_plain_failure(self, cfg):
        plain_failure = terminal_reward(success=False, used_osd=False, deadline_violated=False, cfg=cfg)
        osd_failure = terminal_reward(success=False, used_osd=True, deadline_violated=False, cfg=cfg)
        assert osd_failure < plain_failure

    def test_deadline_violation_is_strongly_penalized(self, cfg):
        r = terminal_reward(success=False, used_osd=False, deadline_violated=True, cfg=cfg)
        # Should be more negative than a bare failure.
        bare_failure = terminal_reward(success=False, used_osd=False, deadline_violated=False, cfg=cfg)
        assert r < bare_failure
        assert r == pytest.approx(cfg.wrong_penalty + cfg.deadline_penalty)

    def test_deadline_violation_is_the_worst_outcome(self, cfg):
        outcomes = [
            terminal_reward(success=True, used_osd=False, deadline_violated=False, cfg=cfg),
            terminal_reward(success=True, used_osd=True, deadline_violated=False, cfg=cfg),
            terminal_reward(success=False, used_osd=False, deadline_violated=False, cfg=cfg),
            terminal_reward(success=False, used_osd=True, deadline_violated=False, cfg=cfg),
            terminal_reward(success=False, used_osd=False, deadline_violated=True, cfg=cfg),
        ]
        assert outcomes[-1] == min(outcomes)

    def test_correct_fast_beats_correct_slow_via_iteration_cost(self, cfg):
        # "Correct but unnecessarily slow -> smaller positive reward":
        # simulate by combining terminal reward with N iteration costs.
        fast_total = terminal_reward(
            success=True, used_osd=False, deadline_violated=False, cfg=cfg
        ) + 2 * continue_step_reward(cfg)
        slow_total = terminal_reward(
            success=True, used_osd=False, deadline_violated=False, cfg=cfg
        ) + 20 * continue_step_reward(cfg)
        assert fast_total > slow_total
        assert fast_total > 0  # still net positive for this cfg

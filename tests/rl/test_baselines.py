"""Unit tests for rl/baselines.py, run against the fast Phase 1 mock
decoder (deterministic, no Stim/ldpc dependency needed for these).
"""

from __future__ import annotations

import pytest

from rl.actions import Action
from rl.baselines import (
    AlwaysContinuePolicy,
    DeterministicStagnationFallbackPolicy,
    FixedIterationStopPolicy,
    compare_policies,
    evaluate_policy,
    run_episode,
)
from rl.config import EnvironmentConfig, MockDecoderConfig, RewardConfig
from rl.environment import QECDecodingEnv
from rl.mock_decoder import MockRelayBPDecoder


def make_env(**env_overrides) -> QECDecodingEnv:
    decoder = MockRelayBPDecoder(MockDecoderConfig())
    env_config = EnvironmentConfig(**env_overrides)
    return QECDecodingEnv(decoder=decoder, env_config=env_config, reward_config=RewardConfig())


class TestAlwaysContinuePolicy:
    def test_always_returns_continue(self):
        policy = AlwaysContinuePolicy()
        # obs/info content shouldn't matter at all for this policy.
        assert policy.select_action(obs=None, info={}) == Action.CONTINUE

    def test_runs_until_guard_or_cap_stops_it(self):
        env = make_env(max_iterations=20, wcet_deadline_iterations=15)
        result = run_episode(env, AlwaysContinuePolicy(), seed=1, physical_error_rate=0.01)
        # Never chose STOP/OSD itself, so termination must have come
        # from the WCET guard (forcing STOP) or the hard cap.
        assert result.final_iteration <= 20
        assert result.decision_steps >= 1


class TestFixedIterationStopPolicy:
    def test_rejects_non_positive_threshold(self):
        with pytest.raises(ValueError):
            FixedIterationStopPolicy(stop_at_iteration=0)

    def test_continues_before_threshold(self):
        policy = FixedIterationStopPolicy(stop_at_iteration=10)
        obs = [0.0, 0.0, 5, 0.0, 0]  # current_iteration = 5
        assert policy.select_action(obs, info={}) == Action.CONTINUE

    def test_stops_at_threshold(self):
        policy = FixedIterationStopPolicy(stop_at_iteration=10)
        obs = [0.0, 0.0, 10, 0.0, 0]  # current_iteration = 10
        assert policy.select_action(obs, info={}) == Action.STOP

    def test_never_uses_osd(self):
        env = make_env(max_iterations=50, wcet_deadline_iterations=45)
        result = run_episode(
            env, FixedIterationStopPolicy(stop_at_iteration=5), seed=1, physical_error_rate=0.01
        )
        assert result.used_osd is False

    def test_stops_at_configured_iteration_when_not_guarded(self):
        env = make_env(max_iterations=50, wcet_deadline_iterations=45)
        result = run_episode(
            env, FixedIterationStopPolicy(stop_at_iteration=5), seed=1, physical_error_rate=0.01
        )
        assert result.final_iteration == 5


class TestDeterministicStagnationFallbackPolicy:
    def test_rejects_non_positive_threshold(self):
        with pytest.raises(ValueError):
            DeterministicStagnationFallbackPolicy(stagnation_threshold=0)

    def test_continues_below_stagnation_threshold(self):
        policy = DeterministicStagnationFallbackPolicy(stagnation_threshold=5)
        obs = [0.0, 0.0, 3, 0.0, 4]  # iterations_since_improvement = 4
        assert policy.select_action(obs, info={}) == Action.CONTINUE

    def test_falls_back_to_osd_at_stagnation_threshold(self):
        policy = DeterministicStagnationFallbackPolicy(stagnation_threshold=5)
        obs = [0.0, 0.0, 3, 0.0, 5]  # iterations_since_improvement = 5
        assert policy.select_action(obs, info={}) == Action.OSD

    def test_can_actually_trigger_osd_in_an_episode(self):
        # A low improvement_threshold + noisy mock trajectory should
        # trigger stagnation at least once across several seeds.
        env = make_env(max_iterations=60, wcet_deadline_iterations=55)
        policy = DeterministicStagnationFallbackPolicy(stagnation_threshold=3)
        used_osd_at_least_once = any(
            run_episode(env, policy, seed=s, physical_error_rate=0.02).used_osd
            for s in range(10)
        )
        assert used_osd_at_least_once


class TestRunEpisode:
    def test_returns_populated_result(self):
        env = make_env()
        result = run_episode(env, AlwaysContinuePolicy(), seed=1, physical_error_rate=0.01)
        assert result.policy_name == "always_continue"
        assert result.seed == 1
        assert result.physical_error_rate == 0.01
        assert result.decision_steps > 0
        assert isinstance(result.success, bool)

    def test_deterministic_given_seed(self):
        env_a = make_env()
        env_b = make_env()
        policy = FixedIterationStopPolicy(stop_at_iteration=8)
        ra = run_episode(env_a, policy, seed=3, physical_error_rate=0.015)
        rb = run_episode(env_b, policy, seed=3, physical_error_rate=0.015)
        assert ra == rb


class TestEvaluatePolicy:
    def test_rejects_empty_seeds(self):
        env = make_env()
        with pytest.raises(ValueError):
            evaluate_policy(env, AlwaysContinuePolicy(), seeds=[], physical_error_rate=0.01)

    def test_aggregates_over_multiple_seeds(self):
        env = make_env(max_iterations=30, wcet_deadline_iterations=25)
        agg = evaluate_policy(
            env,
            FixedIterationStopPolicy(stop_at_iteration=6),
            seeds=[1, 2, 3, 4, 5],
            physical_error_rate=0.01,
        )
        assert agg.n_episodes == 5
        assert 0.0 <= agg.success_rate <= 1.0
        assert agg.mean_final_iteration == pytest.approx(6.0)  # always stops exactly at 6
        assert agg.osd_rate == 0.0  # this policy never uses OSD

    def test_rates_are_within_unit_interval(self):
        env = make_env(max_iterations=30, wcet_deadline_iterations=25)
        agg = evaluate_policy(
            env,
            DeterministicStagnationFallbackPolicy(stagnation_threshold=3),
            seeds=list(range(10)),
            physical_error_rate=0.02,
        )
        for rate in (agg.success_rate, agg.osd_rate, agg.deadline_violation_rate):
            assert 0.0 <= rate <= 1.0


class TestComparePolicies:
    def test_returns_one_result_per_policy_same_order(self):
        env = make_env(max_iterations=30, wcet_deadline_iterations=25)
        policies = [
            AlwaysContinuePolicy(),
            FixedIterationStopPolicy(stop_at_iteration=10),
            DeterministicStagnationFallbackPolicy(stagnation_threshold=4),
        ]
        results = compare_policies(env, policies, seeds=[1, 2, 3], physical_error_rate=0.01)
        assert [r.policy_name for r in results] == [p.name for p in policies]
        assert all(r.n_episodes == 3 for r in results)

    def test_identical_conditions_across_policies(self):
        # Same env/seeds/noise level for every policy in the comparison
        # — this is what makes the AggregateResults comparable at all.
        env = make_env(max_iterations=30, wcet_deadline_iterations=25)
        policies = [AlwaysContinuePolicy(), FixedIterationStopPolicy(stop_at_iteration=10)]
        results = compare_policies(env, policies, seeds=[1, 2, 3], physical_error_rate=0.03)
        assert all(r.physical_error_rate == 0.03 for r in results)
        assert all(r.n_episodes == 3 for r in results)

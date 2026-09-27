"""Unit tests for rl/environment.py, exercised against the MOCK decoder.

These tests validate RL-infrastructure plumbing only (reset/step
contract, WCET guard, action semantics, determinism). They are not QEC
experiments and assert nothing about real decoding performance.
"""

from __future__ import annotations

import numpy as np
import pytest

from rl.actions import Action
from rl.config import EnvironmentConfig, MockDecoderConfig, RewardConfig
from rl.environment import DecoderProtocol, QECDecodingEnv
from rl.mock_decoder import MockRelayBPDecoder


def make_env(**env_overrides) -> QECDecodingEnv:
    decoder = MockRelayBPDecoder(MockDecoderConfig())
    env_config = EnvironmentConfig(**env_overrides)
    return QECDecodingEnv(decoder=decoder, env_config=env_config, reward_config=RewardConfig())


class TestProtocolConformance:
    def test_mock_decoder_satisfies_protocol(self):
        assert isinstance(MockRelayBPDecoder(), DecoderProtocol)


class TestReset:
    def test_reset_returns_observation_and_info(self):
        env = make_env()
        obs, info = env.reset(seed=1)
        assert isinstance(obs, np.ndarray)
        assert isinstance(info, dict)

    def test_observation_shape_matches_state_mode(self):
        oracle_env = make_env(state_mode="oracle")
        obs, _ = oracle_env.reset(seed=1)
        assert obs.shape == (5,)

        realistic_env = make_env(state_mode="realistic")
        obs, _ = realistic_env.reset(seed=1)
        assert obs.shape == (4,)

    def test_reset_starts_at_iteration_zero(self):
        env = make_env()
        obs, info = env.reset(seed=1)
        assert info["current_iteration"] == 0


class TestContinueAction:
    def test_continue_advances_iteration(self):
        env = make_env()
        env.reset(seed=1)
        obs, reward, terminated, truncated, info = env.step(Action.CONTINUE)
        assert info["current_iteration"] == 1
        assert terminated is False

    def test_continue_charges_iteration_cost(self):
        env = make_env()
        env.reset(seed=1)
        _, reward, _, _, _ = env.step(Action.CONTINUE)
        assert reward == pytest.approx(-env.reward_config.iteration_cost)

    def test_repeated_continue_keeps_advancing(self):
        env = make_env(max_iterations=10, wcet_deadline_iterations=10)
        env.reset(seed=1)
        for expected_iter in range(1, 6):
            _, _, terminated, truncated, info = env.step(Action.CONTINUE)
            assert info["current_iteration"] == expected_iter
            assert not terminated
            assert not truncated


class TestStopAction:
    def test_stop_terminates_episode(self):
        env = make_env()
        env.reset(seed=1)
        _, _, terminated, truncated, info = env.step(Action.STOP)
        assert terminated is True
        assert truncated is False
        assert info["success"] in (True, False)

    def test_step_after_stop_raises(self):
        env = make_env()
        env.reset(seed=1)
        env.step(Action.STOP)
        with pytest.raises(RuntimeError):
            env.step(Action.CONTINUE)


class TestOsdAction:
    def test_osd_terminates_and_flags_used_osd(self):
        env = make_env()
        env.reset(seed=1)
        _, _, terminated, truncated, info = env.step(Action.OSD)
        assert terminated is True
        assert info["used_osd"] is True

    def test_osd_success_is_boolean(self):
        env = make_env()
        env.reset(seed=1)
        _, _, _, _, info = env.step(Action.OSD)
        assert isinstance(info["success"], bool)


class TestMaxIterationsEnforced:
    def test_hard_cap_truncates_as_defensive_fallback(self):
        # wcet_deadline_iterations == max_iterations means the guard's
        # own precondition (current_iteration >= deadline, checked
        # *before* stepping) is never true while iteration < deadline,
        # so the third CONTINUE is allowed through, pushes
        # current_iteration to max_iterations, and the defensive
        # hard-cap check in _do_continue truncates the episode as a
        # genuine deadline violation. This exercises that fallback path
        # (see EnvironmentConfig.wcet_deadline_iterations docstring:
        # normally the guard fires strictly before this point, so in
        # practice set wcet_deadline_iterations < max_iterations).
        env = make_env(max_iterations=3, wcet_deadline_iterations=3)
        env.reset(seed=1)
        env.step(Action.CONTINUE)  # iter 1
        env.step(Action.CONTINUE)  # iter 2
        obs, reward, terminated, truncated, info = env.step(Action.CONTINUE)  # iter 3 -> hard cap
        assert info["wcet_guard_activated"] is False
        assert truncated is True
        assert info["deadline_violated"] is True

    def test_guard_set_below_max_prevents_hard_cap_from_ever_triggering(self):
        # The intended configuration: deadline strictly below the hard
        # cap, so the guard always intervenes first.
        env = make_env(max_iterations=5, wcet_deadline_iterations=3)
        env.reset(seed=1)
        env.step(Action.CONTINUE)  # iter 1
        env.step(Action.CONTINUE)  # iter 2
        env.step(Action.CONTINUE)  # iter 3, at deadline
        _, _, terminated, truncated, info = env.step(Action.CONTINUE)  # guard forces STOP
        assert info["wcet_guard_activated"] is True
        assert terminated is True
        assert truncated is False
        assert info["deadline_violated"] is False


class TestWcetGuard:
    def test_guard_forces_stop_at_deadline(self):
        env = make_env(max_iterations=50, wcet_deadline_iterations=2)
        env.reset(seed=1)
        env.step(Action.CONTINUE)  # iter 1
        env.step(Action.CONTINUE)  # iter 2, now at deadline
        _, _, terminated, truncated, info = env.step(Action.CONTINUE)  # requested CONTINUE, should be forced STOP
        assert info["wcet_guard_activated"] is True
        assert info["action_taken"] == int(Action.CONTINUE)  # requested action recorded
        assert terminated is True
        assert not truncated

    def test_guard_does_not_activate_before_deadline(self):
        env = make_env(max_iterations=50, wcet_deadline_iterations=10)
        env.reset(seed=1)
        _, _, _, _, info = env.step(Action.CONTINUE)
        assert info["wcet_guard_activated"] is False

    def test_guard_does_not_interfere_with_explicit_stop(self):
        env = make_env(max_iterations=50, wcet_deadline_iterations=1)
        env.reset(seed=1)
        _, _, terminated, _, info = env.step(Action.STOP)
        assert info["wcet_guard_activated"] is False
        assert terminated is True


class TestDeterminism:
    def test_same_seed_same_trajectory(self):
        env_a = make_env(max_iterations=20, wcet_deadline_iterations=20)
        env_b = make_env(max_iterations=20, wcet_deadline_iterations=20)

        obs_a, _ = env_a.reset(seed=42, physical_error_rate=0.02)
        obs_b, _ = env_b.reset(seed=42, physical_error_rate=0.02)
        np.testing.assert_array_equal(obs_a, obs_b)

        for _ in range(5):
            ra = env_a.step(Action.CONTINUE)
            rb = env_b.step(Action.CONTINUE)
            np.testing.assert_array_equal(ra[0], rb[0])  # obs
            assert ra[1] == rb[1]  # reward
            assert ra[2:] == rb[2:]  # terminated, truncated, info... (info dict compares equal too)

    def test_different_seed_can_differ(self):
        env_a = make_env(max_iterations=20, wcet_deadline_iterations=20)
        env_b = make_env(max_iterations=20, wcet_deadline_iterations=20)

        obs_a, _ = env_a.reset(seed=1, physical_error_rate=0.05)
        obs_b, _ = env_b.reset(seed=2, physical_error_rate=0.05)
        # Not a hard guarantee for all seeds, but with noise_scale > 0
        # and different seeds this should essentially always differ.
        assert not np.array_equal(obs_a, obs_b)


class TestInvalidActions:
    def test_invalid_action_raises_value_error(self):
        env = make_env()
        env.reset(seed=1)
        with pytest.raises(ValueError):
            env.step(99)

    def test_step_before_reset_raises(self):
        env = make_env()
        with pytest.raises(RuntimeError):
            env.step(Action.CONTINUE)

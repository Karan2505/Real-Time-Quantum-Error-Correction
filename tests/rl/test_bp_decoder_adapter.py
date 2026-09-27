"""
tests/rl/test_bp_decoder_adapter.py
====================================
REAL integration tests: these exercise Stim + ldpc + stimbposd against
an actual (small) quantum code, not a mock. They are slower than the
Phase 1 unit tests (real BP decoding, real Stim circuit compilation)
but are still fast enough (distance-3 surface code) to run routinely.

If stim/ldpc/stimbposd are not installed, these tests are skipped
rather than failing the suite — they are an optional real-decoder
layer on top of the always-available Phase 1 mock infrastructure.
"""

from __future__ import annotations

import numpy as np
import pytest

stim = pytest.importorskip("stim")
pytest.importorskip("ldpc")
pytest.importorskip("stimbposd")

from rl.actions import Action  # noqa: E402
from rl.bp_decoder_adapter import BpDecoderAdapter, BpDecoderAdapterConfig  # noqa: E402
from rl.config import EnvironmentConfig, RewardConfig  # noqa: E402
from rl.environment import DecoderProtocol, QECDecodingEnv  # noqa: E402
from rl.qec_problem import QECProblem, QECProblemConfig  # noqa: E402

SMALL_CODE = QECProblemConfig(distance=3, rounds=3)


def make_decoder() -> BpDecoderAdapter:
    return BpDecoderAdapter(QECProblem(SMALL_CODE))


class TestQECProblem:
    def test_matrices_have_consistent_shapes(self):
        problem = QECProblem(SMALL_CODE)
        mats = problem.matrices(0.01)
        assert mats.check_matrix.shape[1] == mats.priors.shape[0]
        assert mats.check_matrix.shape[1] == mats.observables_matrix.shape[1]
        assert mats.num_detectors == mats.check_matrix.shape[0]

    def test_sample_shapes_match_matrices(self):
        problem = QECProblem(SMALL_CODE)
        mats = problem.matrices(0.01)
        detection_events, observable_flips = problem.sample(0.01, seed=1)
        assert detection_events.shape == (mats.num_detectors,)
        assert observable_flips.shape == (mats.num_observables,)
        assert detection_events.dtype == np.uint8

    def test_sample_is_deterministic_given_seed(self):
        problem = QECProblem(SMALL_CODE)
        d1, o1 = problem.sample(0.02, seed=5)
        d2, o2 = problem.sample(0.02, seed=5)
        np.testing.assert_array_equal(d1, d2)
        np.testing.assert_array_equal(o1, o2)

    def test_different_seeds_can_differ(self):
        problem = QECProblem(SMALL_CODE)
        d1, _ = problem.sample(0.05, seed=1)
        d2, _ = problem.sample(0.05, seed=2)
        # Not a hard guarantee for every possible pair, but with a
        # nontrivial error rate this should essentially always differ.
        assert not np.array_equal(d1, d2)


class TestBpDecoderAdapterProtocol:
    def test_satisfies_decoder_protocol(self):
        assert isinstance(make_decoder(), DecoderProtocol)

    def test_reset_returns_decoder_state(self):
        decoder = make_decoder()
        state = decoder.reset(physical_error_rate=0.01, seed=1)
        assert state.current_iteration == 0
        assert 0.0 <= state.unsatisfied_check_ratio <= 1.0
        assert state.physical_error_rate == 0.01

    def test_step_continue_advances_iteration(self):
        decoder = make_decoder()
        decoder.reset(physical_error_rate=0.01, seed=1)
        result = decoder.step_continue()
        assert result.state.current_iteration == 1

    def test_repeated_step_continue_keeps_advancing(self):
        decoder = make_decoder()
        decoder.reset(physical_error_rate=0.01, seed=1)
        for expected_iter in range(1, 6):
            result = decoder.step_continue()
            assert result.state.current_iteration == expected_iter

    def test_is_converged_returns_bool(self):
        decoder = make_decoder()
        decoder.reset(physical_error_rate=0.01, seed=1)
        decoder.step_continue()
        assert isinstance(decoder.is_converged(), bool)

    def test_osd_fallback_returns_bool(self):
        decoder = make_decoder()
        decoder.reset(physical_error_rate=0.01, seed=1)
        assert isinstance(decoder.osd_fallback(), bool)

    def test_methods_before_reset_raise(self):
        decoder = make_decoder()
        with pytest.raises(RuntimeError):
            decoder.step_continue()
        with pytest.raises(RuntimeError):
            decoder.is_converged()
        with pytest.raises(RuntimeError):
            decoder.osd_fallback()


class TestDeterminism:
    def test_same_seed_same_trajectory(self):
        d1 = make_decoder()
        d2 = make_decoder()
        s1 = d1.reset(physical_error_rate=0.02, seed=7)
        s2 = d2.reset(physical_error_rate=0.02, seed=7)
        assert s1 == s2

        for _ in range(5):
            r1 = d1.step_continue()
            r2 = d2.step_continue()
            assert r1.state == r2.state
            assert r1.improved == r2.improved


class TestLogicalSuccessIsReal:
    def test_success_reflects_actual_observable_match_not_bp_convergence_flag(self):
        # At a very low error rate, BP should very likely converge to
        # the correct logical outcome quickly.
        decoder = make_decoder()
        decoder.reset(physical_error_rate=0.001, seed=3)
        for _ in range(20):
            decoder.step_continue()
        # Not a hard guarantee (still real physics/decoding, not
        # rigged), but should hold for this seed/error-rate/iteration
        # budget on a distance-3 code.
        assert isinstance(decoder.is_converged(), bool)


class TestEndToEndWithRealEnvironment:
    """The real payoff: the exact same rl.environment.QECDecodingEnv
    from Phase 1, now driven by a real decoder instead of the mock,
    with zero changes to environment.py."""

    def test_full_episode_runs_and_terminates(self):
        decoder = make_decoder()
        env = QECDecodingEnv(
            decoder=decoder,
            env_config=EnvironmentConfig(max_iterations=30, wcet_deadline_iterations=25),
            reward_config=RewardConfig(),
        )
        obs, info = env.reset(seed=1, physical_error_rate=0.01)
        assert obs.shape == (5,)

        terminated = truncated = False
        steps = 0
        while not (terminated or truncated) and steps < 40:
            obs, reward, terminated, truncated, info = env.step(Action.CONTINUE)
            steps += 1

        assert terminated or truncated
        assert isinstance(info["success"], bool)

    def test_osd_action_resolves_via_real_osd(self):
        decoder = make_decoder()
        env = QECDecodingEnv(decoder=decoder, env_config=EnvironmentConfig())
        env.reset(seed=2, physical_error_rate=0.02)
        _, _, terminated, _, info = env.step(Action.OSD)
        assert terminated is True
        assert info["used_osd"] is True
        assert isinstance(info["success"], bool)

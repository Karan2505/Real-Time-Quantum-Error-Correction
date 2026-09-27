"""
tests/rl/test_baselines_real_decoder.py
=========================================
Confirms the Phase 3 baseline policies and evaluation harness work
unmodified against the REAL decoder (Phase 2's BpDecoderAdapter), not
just the mock — proving compare_policies/evaluate_policy are decoder-
agnostic as designed.

Skipped automatically if stim/ldpc/stimbposd aren't installed.
"""

from __future__ import annotations

import pytest

pytest.importorskip("stim")
pytest.importorskip("ldpc")
pytest.importorskip("stimbposd")

from rl.baselines import (  # noqa: E402
    AlwaysContinuePolicy,
    DeterministicStagnationFallbackPolicy,
    FixedIterationStopPolicy,
    compare_policies,
)
from rl.bp_decoder_adapter import BpDecoderAdapter  # noqa: E402
from rl.config import EnvironmentConfig  # noqa: E402
from rl.environment import QECDecodingEnv  # noqa: E402
from rl.qec_problem import QECProblem, QECProblemConfig  # noqa: E402

SMALL_CODE = QECProblemConfig(distance=3, rounds=3)


def make_real_env(**env_overrides) -> QECDecodingEnv:
    decoder = BpDecoderAdapter(QECProblem(SMALL_CODE))
    return QECDecodingEnv(decoder=decoder, env_config=EnvironmentConfig(**env_overrides))


class TestBaselinesAgainstRealDecoder:
    def test_compare_policies_runs_against_real_decoder(self):
        env = make_real_env(max_iterations=30, wcet_deadline_iterations=25)
        policies = [
            AlwaysContinuePolicy(),
            FixedIterationStopPolicy(stop_at_iteration=10),
            DeterministicStagnationFallbackPolicy(stagnation_threshold=4),
        ]
        results = compare_policies(env, policies, seeds=[1, 2, 3, 4], physical_error_rate=0.01)

        assert len(results) == 3
        for r in results:
            assert r.n_episodes == 4
            assert 0.0 <= r.success_rate <= 1.0
            assert r.mean_final_iteration > 0

    def test_fixed_stop_matches_configured_iteration_on_real_decoder(self):
        # Sanity check that the real decoder path exercises the same
        # policy logic correctly (not just "runs without crashing").
        env = make_real_env(max_iterations=30, wcet_deadline_iterations=25)
        from rl.baselines import run_episode

        result = run_episode(
            env, FixedIterationStopPolicy(stop_at_iteration=7), seed=1, physical_error_rate=0.01
        )
        assert result.final_iteration == 7
        assert result.used_osd is False

"""
rl/compare_baselines.py
=========================
Prints a side-by-side comparison table of the three Phase 3 baseline
policies (rl/baselines.py) at a given physical error rate, over
multiple seeds. Works against either decoder backend.

This is the actual scientific comparison scaffold your project spec
calls for (Section 14/16) — not a plumbing demo like rl/demo_run.py.
Numbers from the mock decoder (default) are still infrastructure-only;
pass --real-decoder for numbers against the real (but not-yet-target-
code, not-yet-Relay-BP) Phase 2 pipeline.

Usage
-----
    python -m rl.compare_baselines
    python -m rl.compare_baselines --real-decoder --physical-error-rate 0.01 --n-seeds 20
    python -m rl.compare_baselines --physical-error-rate 0.005 --physical-error-rate 0.02
"""

from __future__ import annotations

import argparse

from rl.baselines import (
    AggregateResult,
    AlwaysContinuePolicy,
    DeterministicStagnationFallbackPolicy,
    FixedIterationStopPolicy,
    compare_policies,
)
from rl.config import EnvironmentConfig, MockDecoderConfig, RewardConfig
from rl.environment import QECDecodingEnv
from rl.mock_decoder import MockRelayBPDecoder


def print_table(results: list[AggregateResult], physical_error_rate: float) -> None:
    print(f"\n=== physical_error_rate = {physical_error_rate} (n={results[0].n_episodes} seeds) ===")
    header = (
        f"{'policy':<32} {'success%':>9} {'mean_iter':>10} {'osd%':>7} "
        f"{'deadline%':>10} {'guard_acts':>11} {'mean_reward':>12}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r.policy_name:<32} {r.success_rate * 100:>8.1f}% {r.mean_final_iteration:>10.1f} "
            f"{r.osd_rate * 100:>6.1f}% {r.deadline_violation_rate * 100:>9.1f}% "
            f"{r.mean_wcet_guard_activations:>11.2f} {r.mean_reward:>12.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physical-error-rate", type=float, action="append", default=None)
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--max-iterations", type=int, default=50)
    parser.add_argument("--wcet-deadline-iterations", type=int, default=40)
    parser.add_argument("--fixed-stop-at", type=int, default=15)
    parser.add_argument("--stagnation-threshold", type=int, default=5)
    parser.add_argument("--real-decoder", action="store_true")
    args = parser.parse_args()

    error_rates = args.physical_error_rate or [0.01]

    if args.real_decoder:
        from rl.bp_decoder_adapter import BpDecoderAdapter
        from rl.qec_problem import QECProblem

        print(">>> REAL decoder backend (Stim surface code + ldpc BP/OSD) <<<")
        decoder = BpDecoderAdapter(QECProblem())
    else:
        print(">>> MOCK decoder backend (synthetic, infrastructure-only) <<<")
        decoder = MockRelayBPDecoder(MockDecoderConfig())

    env = QECDecodingEnv(
        decoder=decoder,
        env_config=EnvironmentConfig(
            max_iterations=args.max_iterations,
            wcet_deadline_iterations=args.wcet_deadline_iterations,
        ),
        reward_config=RewardConfig(),
    )

    policies = [
        AlwaysContinuePolicy(),
        FixedIterationStopPolicy(stop_at_iteration=args.fixed_stop_at),
        DeterministicStagnationFallbackPolicy(stagnation_threshold=args.stagnation_threshold),
    ]
    seeds = list(range(args.n_seeds))

    for per in error_rates:
        results = compare_policies(env, policies, seeds=seeds, physical_error_rate=per)
        print_table(results, per)

    print(
        "\nReminder: these are deterministic baselines, not RL — no learning happened here. "
        "They exist so a future trained policy has something honest to beat."
    )


if __name__ == "__main__":
    main()

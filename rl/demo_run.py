"""
rl/demo_run.py
==============
NOT AN RL AGENT. This script exists only so a human can visually watch
the Phase-1 environment (rl/environment.py + rl/mock_decoder.py) run
end-to-end, one decision at a time, before any learning agent exists.

It drives the environment with a trivial, hard-coded PLACEHOLDER policy
("keep going until the mock decoder says it has converged, or the
stagnation counter gets too high, then stop") — this is NOT reinforcement
learning. It does not learn, adapt, or optimize anything. It is the
RL-free equivalent of manually pressing "next" to prove the game board
works before anyone has trained a player.

Do not use any printed numbers here as evidence of RL performance, QEC
performance, or anything scientific. This is a plumbing demo only.

Usage
-----
    python -m rl.demo_run
    python -m rl.demo_run --seed 7 --physical-error-rate 0.02
    python -m rl.demo_run --episodes 5
    python -m rl.demo_run --real-decoder --physical-error-rate 0.01

``--real-decoder`` swaps the synthetic mock for
rl.bp_decoder_adapter.BpDecoderAdapter — real Stim-sampled syndromes on
a real (distance-3 surface code) quantum code, real BP decoding via
ldpc.BpDecoder, real OSD via ldpc.BpOsdDecoder, and success/failure
judged by a genuine logical-observable check. This requires
`pip install -r rl/requirements-real-decoder.txt`. It is still NOT
Relay-BP (plain BP instead) and still NOT the project's [[144,12,12]]
target code (a smaller surface code instead) — see
rl/bp_decoder_adapter.py and rl/qec_problem.py docstrings for exactly
what that does and doesn't mean.
"""

from __future__ import annotations

import argparse

from rl.actions import Action
from rl.config import EnvironmentConfig, MockDecoderConfig, RewardConfig
from rl.environment import QECDecodingEnv
from rl.mock_decoder import MockRelayBPDecoder


def placeholder_policy(current_iteration: int, iterations_since_improvement: int) -> Action:
    """A trivial, hard-coded decision rule. NOT RL.

    Real logic (for demo purposes only):
      - If the mock decoder has stagnated for a while, try OSD once.
      - Otherwise keep going (the environment's own WCET guard and
        max_iterations cap are what actually keep this safe/bounded).

    A trained DQN (Phase 4) will replace this function's role entirely,
    learning its own policy instead of following this fixed rule.
    """
    if iterations_since_improvement >= 5:
        return Action.OSD
    return Action.CONTINUE


def run_episode(env: QECDecodingEnv, seed: int, physical_error_rate: float, verbose: bool = True) -> dict:
    obs, info = env.reset(seed=seed, physical_error_rate=physical_error_rate)
    total_reward = 0.0
    step_count = 0

    if verbose:
        print(f"\n=== Episode start (seed={seed}, physical_error_rate={physical_error_rate}) ===")
        print(f"{'iter':>4} | {'action':>8} | {'ratio':>7} | {'dLLR':>7} | "
              f"{'stagn':>5} | {'reward':>7} | notes")

    iterations_since_improvement = 0
    terminated = truncated = False

    while not (terminated or truncated):
        current_iteration = info["current_iteration"]
        action = placeholder_policy(current_iteration, iterations_since_improvement)

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step_count += 1

        # obs layout depends on state_mode; oracle mode: [ratio, dLLR, iter, per, stagn]
        ratio = obs[0]
        d_llr = obs[1]
        iterations_since_improvement = int(obs[-1])

        note = ""
        if info["wcet_guard_activated"]:
            note = "WCET GUARD forced STOP"
        elif terminated or truncated:
            outcome = "SUCCESS" if info["success"] else "FAILURE"
            osd_tag = " (via OSD)" if info["used_osd"] else ""
            deadline_tag = " [DEADLINE VIOLATION]" if info["deadline_violated"] else ""
            note = f"episode end: {outcome}{osd_tag}{deadline_tag}"

        if verbose:
            print(f"{info['current_iteration']:>4} | {Action(info['action_taken']).name:>8} | "
                  f"{ratio:>7.4f} | {d_llr:>7.4f} | {iterations_since_improvement:>5} | "
                  f"{reward:>7.3f} | {note}")

    if verbose:
        print(f"--- Episode end: {step_count} steps, total_reward={total_reward:.3f}, "
              f"success={info['success']}, used_osd={info['used_osd']} ---")

    return {
        "steps": step_count,
        "total_reward": total_reward,
        "success": info["success"],
        "used_osd": info["used_osd"],
        "deadline_violated": info["deadline_violated"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0, help="Base RNG seed.")
    parser.add_argument("--physical-error-rate", type=float, default=0.01)
    parser.add_argument("--episodes", type=int, default=1, help="How many episodes to run.")
    parser.add_argument(
        "--real-decoder",
        action="store_true",
        help="Use the real BP/OSD decoder adapter (rl.bp_decoder_adapter) instead of the mock. "
        "Requires: pip install -r rl/requirements-real-decoder.txt",
    )
    args = parser.parse_args()

    print(__doc__)

    if args.real_decoder:
        from rl.bp_decoder_adapter import BpDecoderAdapter
        from rl.qec_problem import QECProblem

        print(">>> Using REAL decoder: Stim surface code + ldpc.BpDecoder/BpOsdDecoder <<<\n")
        decoder = BpDecoderAdapter(QECProblem())
    else:
        print(">>> Using MOCK decoder (rl.mock_decoder) — synthetic, infrastructure-only <<<\n")
        decoder = MockRelayBPDecoder(MockDecoderConfig())

    env = QECDecodingEnv(
        decoder=decoder,
        env_config=EnvironmentConfig(max_iterations=50, wcet_deadline_iterations=40),
        reward_config=RewardConfig(),
    )

    results = []
    for i in range(args.episodes):
        result = run_episode(
            env,
            seed=args.seed + i,
            physical_error_rate=args.physical_error_rate,
            verbose=True,
        )
        results.append(result)

    if args.episodes > 1:
        n_success = sum(r["success"] for r in results)
        n_osd = sum(r["used_osd"] for r in results)
        avg_steps = sum(r["steps"] for r in results) / len(results)
        print(f"\n=== Summary over {args.episodes} episodes (PLACEHOLDER policy, MOCK decoder) ===")
        print(f"success rate: {n_success}/{args.episodes}  |  OSD used: {n_osd}/{args.episodes}  |  "
              f"avg steps: {avg_steps:.1f}")
        print("Reminder: these numbers describe plumbing behavior on a synthetic mock decoder "
              "under a hand-coded rule, not RL performance or QEC research results.")


if __name__ == "__main__":
    main()

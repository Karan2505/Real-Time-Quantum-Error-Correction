"""
rl/config.py
============
Configuration objects for the RL controller infrastructure.

All tunable numeric parameters used by the environment, reward function,
and mock decoder live here so that later reward-weight sensitivity
sweeps and environment-parameter experiments (Phase 5) do not require
touching implementation code.

Every default value below is a deliberately simple, documented
PLACEHOLDER chosen to make Phase-1 infrastructure testing possible.
None of them are derived from QEC literature, a fitted noise model, or
any experiment, and none should be quoted as scientific defaults in a
paper or report. They exist purely so the environment/reward/mock
decoder can be exercised end-to-end and unit tested.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MockDecoderConfig:
    """Parameters controlling the synthetic mock decoder's iteration
    dynamics. See rl/mock_decoder.py for the MOCK/PLACEHOLDER label and
    full rationale.

    Attributes
    ----------
    decay_factor:
        Multiplicative shrinkage applied to ``unsatisfied_check_ratio``
        each CONTINUE step, in [0, 1). Larger values -> slower synthetic
        convergence. Purely synthetic; not derived from any BP model.
    noise_scale:
        Standard deviation of Gaussian noise added to the ratio update
        each step, to avoid a perfectly deterministic monotonic curve
        (which would make the mock trivially easy for any policy and
        useless for exercising stagnation-handling logic).
    improvement_threshold:
        Minimum decrease in ``unsatisfied_check_ratio`` between
        iterations that counts as "improvement". Smaller decreases
        increment ``iterations_since_improvement``.
    convergence_threshold:
        ``unsatisfied_check_ratio`` at or below this value is treated as
        a converged/successful synthetic solution when the episode
        stops.
    initial_ratio_scale:
        Scales ``physical_error_rate`` into a starting
        ``unsatisfied_check_ratio``. Purely synthetic.
    """

    decay_factor: float = 0.85
    noise_scale: float = 0.01
    improvement_threshold: float = 0.002
    convergence_threshold: float = 0.02
    initial_ratio_scale: float = 4.0

    def __post_init__(self) -> None:
        if not (0.0 <= self.decay_factor < 1.0):
            raise ValueError("decay_factor must be in [0, 1).")
        if self.noise_scale < 0.0:
            raise ValueError("noise_scale must be >= 0.")
        if self.improvement_threshold < 0.0:
            raise ValueError("improvement_threshold must be >= 0.")
        if not (0.0 <= self.convergence_threshold <= 1.0):
            raise ValueError("convergence_threshold must be in [0, 1].")
        if self.initial_ratio_scale <= 0.0:
            raise ValueError("initial_ratio_scale must be > 0.")


@dataclass(frozen=True)
class EnvironmentConfig:
    """Episode-level configuration for :class:`rl.environment.QECDecodingEnv`.

    Attributes
    ----------
    max_iterations:
        Hard cap on CONTINUE steps per episode. Reaching this cap
        without the agent (or the WCET guard) choosing STOP/OSD ends
        the episode via truncation, flagged as a deadline violation
        (see ``deadline_violated`` in step info).
    wcet_deadline_iterations:
        Iteration count at or beyond which the deterministic WCET guard
        overrides a requested CONTINUE action with a forced STOP. Must
        be <= max_iterations. This models "RL is an optimization layer,
        not the real-time safety mechanism" (see project spec, WCET
        section): the guard, not the policy, is responsible for never
        letting the episode reach max_iterations while still running.
        This is a configurable abstraction for Phase 1 development, not
        a measured or formally proven real-time guarantee.
    default_physical_error_rate:
        Physical error rate used by ``reset()`` when the caller does not
        supply one explicitly.
    state_mode:
        "oracle" or "realistic". See rl/state.py for the distinction.
        Default is "oracle" because Phase 1 is infrastructure testing,
        not a claim about deployability.
    """

    max_iterations: int = 50
    wcet_deadline_iterations: int = 40
    default_physical_error_rate: float = 0.01
    state_mode: str = "oracle"

    def __post_init__(self) -> None:
        if self.max_iterations <= 0:
            raise ValueError("max_iterations must be > 0.")
        if not (0 < self.wcet_deadline_iterations <= self.max_iterations):
            raise ValueError(
                "wcet_deadline_iterations must be in (0, max_iterations]."
            )
        if not (0.0 <= self.default_physical_error_rate <= 1.0):
            raise ValueError("default_physical_error_rate must be in [0, 1].")
        if self.state_mode not in ("oracle", "realistic"):
            raise ValueError('state_mode must be "oracle" or "realistic".')


@dataclass(frozen=True)
class RewardConfig:
    """Coefficients for the terminal/step reward function.

    None of these values are the result of a sensitivity analysis or a
    fitted scientific study — they are simple, documented placeholders
    intended to produce the qualitative reward ordering described in
    the project spec (correct+fast > correct+slow > wrong; deadline
    violation is the worst outcome). Section 13 of the project spec
    calls for reward-weight sensitivity experiments before any of these
    numbers are used to support a research claim; that work happens in
    a later phase, not here.

    Attributes
    ----------
    correct_reward:
        Terminal reward when the accepted correction is judged correct
        (i.e. the synthetic/real decoding is considered converged).
    wrong_penalty:
        Terminal reward (negative) when the accepted correction is
        judged incorrect.
    iteration_cost:
        Cost subtracted for each CONTINUE step taken, so unnecessary
        iterations are discouraged even on an eventually-correct run.
    osd_cost:
        Additional cost subtracted whenever OSD is invoked (on top of
        the correct/wrong terminal reward), representing OSD's higher
        computational cost relative to plain BP.
    deadline_penalty:
        Large negative reward applied when the episode ends via a true
        deadline violation (i.e. the WCET guard's own hard cutoff was
        reached without having stopped). Stored as a negative number;
        applied directly (not via ``abs``) in reward.py.
    """

    correct_reward: float = 10.0
    wrong_penalty: float = -10.0
    iteration_cost: float = 0.1
    osd_cost: float = 2.0
    deadline_penalty: float = -50.0

    def __post_init__(self) -> None:
        if self.iteration_cost < 0.0:
            raise ValueError(
                "iteration_cost is subtracted internally; store it as a "
                "non-negative magnitude."
            )
        if self.osd_cost < 0.0:
            raise ValueError(
                "osd_cost is subtracted internally; store it as a "
                "non-negative magnitude."
            )
        if self.deadline_penalty > 0.0:
            raise ValueError("deadline_penalty should be <= 0.")

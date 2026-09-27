"""
rl/bp_decoder_adapter.py
=========================
REAL (not mocked) iterative decoder adapter, satisfying the same
``DecoderProtocol`` as ``rl.mock_decoder.MockRelayBPDecoder``
(rl/environment.py never needs to change to use this).

What is real here, precisely
-----------------------------
* The syndrome is a real Stim-sampled detection-event vector for a
  real quantum code (see rl/qec_problem.py), not a synthetic array.
* The decoder is ``ldpc.BpDecoder`` — real, published belief
  propagation (Roffe et al.), not this project's invention.
* The OSD fallback is ``ldpc.BpOsdDecoder`` — real, published
  Ordered Statistics Decoding, not a mock.
* Success/failure is judged by comparing the decoded correction's
  effect on the code's logical observables against the *actual*
  sampled observable flips — a genuine logical-error check, not a
  synthetic unsatisfied-ratio threshold.

What is NOT yet real / not yet matched to the project spec
-------------------------------------------------------------
* This is plain BP + OSD, not literally *Relay-BP* (which adds
  disordered-memory-strength ensembling on top of BP; see
  arXiv:2506.01779). The real ``relay_bp`` package implements that,
  but requires a Rust toolchain to build on this platform — see the
  Phase 2 scoping notes. Do not call this file's decoder "Relay-BP" in
  any report; call it "BP (ldpc)".
* The underlying code is Stim's rotated surface code (see
  rl/qec_problem.py), not yet the project's [[144,12,12]] target code.

Per-iteration state via repeated ``max_iter=k`` calls
--------------------------------------------------------
``ldpc.BpDecoder`` exposes a single-shot ``decode(syndrome)`` call
governed by a fixed ``max_iter``; it does not expose a "run one more
iteration, keeping prior state" method. Because BP with a fixed
parallel schedule and fixed priors is deterministic, calling
``decode()`` fresh with ``max_iter=k`` for increasing ``k`` reproduces
the exact state the decoder would be in after k iterations of a single
continuous run. This adapter uses that approach: each
``step_continue()`` re-decodes from scratch with ``max_iter`` set to
the new iteration count. This is more compute per step than a decoder
with a true incremental-iteration API would need (each step re-does
all prior iterations, not just the new one), which matters for
runtime/throughput experiments later but does not change what state is
observed — see Phase 2 scoping notes for why fine-grained,
non-recomputing control was deferred.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from ldpc import BpDecoder, BpOsdDecoder

from rl.qec_problem import QECMatrices, QECProblem
from rl.state import DecoderState


@dataclass(frozen=True)
class BpDecoderAdapterConfig:
    """Tunables for the BP/OSD decoder itself (distinct from
    :class:`rl.qec_problem.QECProblemConfig`, which shapes the QEC
    instance being decoded).

    Attributes
    ----------
    bp_method:
        'minimum_sum' or 'product_sum'; passed straight to
        ``ldpc.BpDecoder``/``BpOsdDecoder``.
    improvement_threshold:
        Minimum drop in ``unsatisfied_check_ratio`` between iterations
        that counts as "improvement", for
        ``iterations_since_improvement`` bookkeeping — same semantics
        as ``MockDecoderConfig.improvement_threshold``.
    osd_method / osd_order:
        Passed straight to ``ldpc.BpOsdDecoder``. See its docstring.
    osd_max_bp_iter:
        ``max_iter`` used when constructing the fallback
        ``BpOsdDecoder`` in ``osd_fallback()``.
    """

    bp_method: str = "minimum_sum"
    improvement_threshold: float = 0.002
    osd_method: str = "osd_cs"
    osd_order: int = 10
    osd_max_bp_iter: int = 30

    def __post_init__(self) -> None:
        if self.bp_method not in ("minimum_sum", "product_sum"):
            raise ValueError('bp_method must be "minimum_sum" or "product_sum".')
        if self.improvement_threshold < 0.0:
            raise ValueError("improvement_threshold must be >= 0.")
        if self.osd_order < 0:
            raise ValueError("osd_order must be >= 0.")
        if self.osd_max_bp_iter <= 0:
            raise ValueError("osd_max_bp_iter must be > 0.")


@dataclass
class IterationResult:
    """Return value of a single ``step_continue()`` call. Same shape as
    ``rl.mock_decoder.MockIterationResult`` by convention (duck-typed
    by ``rl.environment``, which only needs ``.state``)."""

    state: DecoderState
    improved: bool


class BpDecoderAdapter:
    """REAL (not mocked) decoder backend for
    :class:`rl.environment.QECDecodingEnv`, implementing the same
    ``DecoderProtocol`` as :class:`rl.mock_decoder.MockRelayBPDecoder`.

    See module docstring for exactly what is and is not real here.
    """

    def __init__(
        self,
        qec_problem: QECProblem | None = None,
        config: BpDecoderAdapterConfig | None = None,
    ) -> None:
        self._problem = qec_problem or QECProblem()
        self._config = config or BpDecoderAdapterConfig()

        self._mats: QECMatrices | None = None
        self._syndrome: np.ndarray | None = None
        self._true_observable_flips: np.ndarray | None = None
        self._physical_error_rate: float = 0.0

        self._iteration: int = 0
        self._iterations_since_improvement: int = 0
        self._prev_unsatisfied_ratio: float = 1.0
        self._prev_llrs: np.ndarray | None = None

        self._last_correction: np.ndarray | None = None
        self._last_bp_converged: bool = False
        self._initialized = False

    @property
    def config(self) -> BpDecoderAdapterConfig:
        return self._config

    @property
    def current_iteration(self) -> int:
        return self._iteration

    def reset(self, physical_error_rate: float, seed: int) -> DecoderState:
        """Sample a real QEC instance at this noise level and return
        its initial (pre-decoding) state."""
        if not (0.0 <= physical_error_rate <= 1.0):
            raise ValueError("physical_error_rate must be in [0, 1].")

        self._mats = self._problem.matrices(physical_error_rate)
        self._syndrome, self._true_observable_flips = self._problem.sample(
            physical_error_rate, seed
        )
        self._physical_error_rate = physical_error_rate

        self._iteration = 0
        self._iterations_since_improvement = 0
        # Before any BP iteration, "unsatisfied_check_ratio" is simply
        # the fraction of syndrome bits that are already 1 (checks
        # violated by the raw syndrome, with zero correction applied).
        self._prev_unsatisfied_ratio = float(np.mean(self._syndrome)) if self._syndrome.size else 0.0
        self._prev_llrs = None
        self._last_correction = np.zeros(self._mats.check_matrix.shape[1], dtype=np.uint8)
        self._last_bp_converged = False
        self._initialized = True

        return DecoderState(
            unsatisfied_check_ratio=self._prev_unsatisfied_ratio,
            delta_LLR=0.0,
            current_iteration=0,
            physical_error_rate=physical_error_rate,
            iterations_since_improvement=0,
        )

    def step_continue(self) -> IterationResult:
        """Advance by one BP iteration (see module docstring: this
        re-decodes from scratch with ``max_iter = new iteration
        count``, which is state-equivalent to true incremental
        stepping for a deterministic fixed-schedule BP decoder)."""
        self._require_initialized()

        self._iteration += 1
        decoder = BpDecoder(
            self._mats.check_matrix,
            error_channel=list(self._mats.priors),
            max_iter=self._iteration,
            bp_method=self._config.bp_method,
        )
        correction = decoder.decode(self._syndrome)
        current_llrs = np.asarray(decoder.log_prob_ratios, dtype=np.float64)

        implied_syndrome = (self._mats.check_matrix @ correction) % 2
        unsatisfied_ratio = float(
            np.count_nonzero(implied_syndrome != self._syndrome) / max(self._syndrome.size, 1)
        )

        if self._prev_llrs is None:
            delta_llr = float(np.mean(np.abs(current_llrs)))
        else:
            delta_llr = float(np.mean(np.abs(current_llrs - self._prev_llrs)))

        improvement = self._prev_unsatisfied_ratio - unsatisfied_ratio
        improved = improvement >= self._config.improvement_threshold
        self._iterations_since_improvement = (
            0 if improved else self._iterations_since_improvement + 1
        )

        self._prev_unsatisfied_ratio = unsatisfied_ratio
        self._prev_llrs = current_llrs
        self._last_correction = correction
        self._last_bp_converged = bool(decoder.converge)

        state = DecoderState(
            unsatisfied_check_ratio=unsatisfied_ratio,
            delta_LLR=delta_llr,
            current_iteration=self._iteration,
            physical_error_rate=self._physical_error_rate,
            iterations_since_improvement=self._iterations_since_improvement,
        )
        return IterationResult(state=state, improved=improved)

    def is_converged(self) -> bool:
        """Real logical-error check: does the current correction's
        predicted effect on the logical observables match what
        actually happened (per the Stim sample), regardless of whether
        BP's own internal convergence flag fired?

        This is a genuine correctness check, not a synthetic proxy —
        it is exactly the standard QEC "did the decoder get the
        logical outcome right" criterion.
        """
        self._require_initialized()
        return self._logical_success(self._last_correction)

    def osd_fallback(self) -> bool:
        """Real OSD fallback via ``ldpc.BpOsdDecoder`` (not mocked)."""
        self._require_initialized()
        decoder = BpOsdDecoder(
            self._mats.check_matrix,
            error_channel=list(self._mats.priors),
            max_iter=self._config.osd_max_bp_iter,
            bp_method=self._config.bp_method,
            osd_method=self._config.osd_method,
            osd_order=self._config.osd_order,
        )
        correction = decoder.decode(self._syndrome)
        self._last_correction = correction
        return self._logical_success(correction)

    def _logical_success(self, correction: np.ndarray) -> bool:
        predicted_obs = (self._mats.observables_matrix @ correction) % 2
        predicted_obs = np.asarray(predicted_obs, dtype=np.uint8).flatten()
        true_obs = np.asarray(self._true_observable_flips, dtype=np.uint8).flatten()
        return bool(np.array_equal(predicted_obs, true_obs))

    def _require_initialized(self) -> None:
        if not self._initialized:
            raise RuntimeError(
                "BpDecoderAdapter.reset() must be called before "
                "step_continue()/is_converged()/osd_fallback()."
            )

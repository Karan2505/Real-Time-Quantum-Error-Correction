"""
rl/qec_problem.py
==================
REAL (not mocked) QEC problem generation.

Uses Stim to build a noisy quantum circuit for a real quantum
error-correcting code, extracts its detector error model (DEM), and
converts that DEM into the matrices a BP decoder needs (check matrix,
observables matrix, per-mechanism error priors) via ``stimbposd``'s
``detector_error_model_to_check_matrices``.

Code choice: rotated surface code, not yet the [[144,12,12]] bivariate
bicycle ("gross") code
-----------------------------------------------------------------------
The project's target code is the [[144,12,12]] qLDPC code, and Stim's
own circuit generator does not build that code directly — the
reference implementation for it lives in the ``relay-bp`` package's
bundled test-circuit data (see ``relay_bp.stim.testdata``), which
requires the Rust toolchain to install (see Phase 2 scoping notes).

Until that's available, this module uses Stim's built-in
``surface_code:rotated_memory_z`` generator instead. This is a
legitimate, well-studied, real quantum code — not a placeholder — but
it is explicitly NOT the [[144,12,12]] target code from the project
spec. Results produced against this code characterize RL-infrastructure
behavior against a real (small) QEC instance; they are not evidence
about the [[144,12,12]] code specifically, and must not be reported as
such. Swapping to the real target code later means changing only
``QECProblemConfig``/``QECProblem`` (specifically, building the check
matrix from the gross code's own parity-check data instead of a Stim
circuit) — nothing downstream (the BP decoder adapter, the RL
environment) needs to change, because both consume the same
``(check_matrix, observables_matrix, priors)`` triple regardless of
where it came from.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import stim
from scipy.sparse import csc_matrix
from stimbposd import detector_error_model_to_check_matrices


@dataclass(frozen=True)
class QECProblemConfig:
    """Parameters defining one QEC problem instance (a fixed code +
    noise-model shape; the actual noise *level* is supplied separately
    per-episode as ``physical_error_rate``, matching
    ``EnvironmentConfig``/``DecoderState``).

    Attributes
    ----------
    code_task:
        Stim's circuit-generator task name. Default is a real, small,
        well-studied code — NOT the project's [[144,12,12]] target
        code (see module docstring).
    distance:
        Code distance.
    rounds:
        Number of syndrome-extraction rounds simulated per instance.
    decompose_errors:
        Passed to ``circuit.detector_error_model()``. Keeping this True
        is what allows ``stimbposd`` to build a usable (mostly
        graphlike) check matrix; see stimbposd's docs for the
        ``allow_undecomposed_hyperedges`` fallback for any residual
        hyperedges.
    """

    code_task: str = "surface_code:rotated_memory_z"
    distance: int = 3
    rounds: int = 3
    decompose_errors: bool = True

    def __post_init__(self) -> None:
        if self.distance < 3 or self.distance % 2 == 0:
            raise ValueError("distance must be an odd integer >= 3 for this code family.")
        if self.rounds < 1:
            raise ValueError("rounds must be >= 1.")


@dataclass(frozen=True)
class QECMatrices:
    """The matrices a BP/OSD decoder needs, plus the circuit they came
    from (kept for provenance/debugging, not required by the decoder).
    """

    check_matrix: csc_matrix
    observables_matrix: csc_matrix
    priors: np.ndarray
    num_detectors: int
    num_observables: int
    circuit: stim.Circuit


class QECProblem:
    """Builds and caches real Stim circuits / DEM-derived matrices for
    a fixed :class:`QECProblemConfig`, varying only the noise level
    (``physical_error_rate``) and the sampling seed per call — this
    matches how :class:`rl.environment.QECDecodingEnv` calls
    ``reset(physical_error_rate=..., seed=...)`` once per episode.

    Circuit/DEM construction is the expensive part; it is cached per
    distinct ``physical_error_rate`` (rounded to avoid float-key churn)
    so repeated episodes at the same noise level (typical for a fixed
    noise-sweep point) don't rebuild it from scratch.
    """

    def __init__(self, config: QECProblemConfig | None = None) -> None:
        self._config = config or QECProblemConfig()

    @property
    def config(self) -> QECProblemConfig:
        return self._config

    def matrices(self, physical_error_rate: float) -> QECMatrices:
        """Build (or fetch from cache) the check/observables matrices
        and priors for this code at the given noise level."""
        key = round(float(physical_error_rate), 8)
        return self._matrices_cached(key)

    @lru_cache(maxsize=64)
    def _matrices_cached(self, physical_error_rate_key: float) -> QECMatrices:
        cfg = self._config
        circuit = stim.Circuit.generated(
            cfg.code_task,
            distance=cfg.distance,
            rounds=cfg.rounds,
            after_clifford_depolarization=physical_error_rate_key,
        )
        dem = circuit.detector_error_model(decompose_errors=cfg.decompose_errors)
        mats = detector_error_model_to_check_matrices(dem, allow_undecomposed_hyperedges=True)
        return QECMatrices(
            check_matrix=mats.check_matrix,
            observables_matrix=mats.observables_matrix,
            priors=np.asarray(mats.priors, dtype=np.float64),
            num_detectors=mats.check_matrix.shape[0],
            num_observables=mats.observables_matrix.shape[0],
            circuit=circuit,
        )

    def sample(
        self, physical_error_rate: float, seed: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Draw one real syndrome + its true logical-observable outcome.

        Returns
        -------
        (detection_events, observable_flips):
            ``detection_events``: 1D ``uint8`` array, length
            ``num_detectors`` — this is the real syndrome to decode.
            ``observable_flips``: 1D ``uint8`` array, length
            ``num_observables`` — ground truth for whether each logical
            observable was actually flipped by the sampled physical
            error. This is what makes success/failure checking in
            :class:`rl.bp_decoder_adapter.BpDecoderAdapter` a genuine
            logical-error check rather than a synthetic proxy.
        """
        mats = self.matrices(physical_error_rate)
        sampler = mats.circuit.compile_detector_sampler(seed=seed)
        detection_events, observable_flips = sampler.sample(
            shots=1, separate_observables=True
        )
        return (
            detection_events[0].astype(np.uint8),
            observable_flips[0].astype(np.uint8),
        )

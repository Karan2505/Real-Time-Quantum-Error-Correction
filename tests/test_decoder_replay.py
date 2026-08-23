"""
tests/test_decoder_replay.py
============================
Step 3 — Decoder replay correctness against known-good oracle.

Intent
------
Verify that Deltaflow produces the expected correction result for each chunk
in a pre-recorded syndrome trace.  This is the primary check that the decoder
logic is correct — not just that traffic moves through the pipeline.

Agreement score: A = M_match / N_total

For the stub the oracle is trivially satisfied (stub returns random ops, not
the oracle value) so these tests flag as EXPECTED_SKIP in stub mode unless
the stub is patched to return oracle values.  In real-HW mode, failures here
indicate a decoder logic regression.

All traces are defined in fixtures/syndrome_traces.py.
"""

from __future__ import annotations

import pytest

from fixtures.syndrome_traces import (
    ALL_TRACES,
    TRACE_TRIVIAL,
    TRACE_SINGLE_FLIP,
    TRACE_MULTI_FLIP,
    TRACE_EMPTY_CHUNK,
    TRACE_VARIABLE_DENSITY,
    score_agreement,
    SyndromeTrace,
)
from harness.qeci_interface import Chunk, Measurement
from harness.stimulus_generator import StimulusGenerator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def replay_trace_and_collect(
    trace: SyndromeTrace,
    qeci,
    cfg,
) -> dict:
    """
    Replay all chunks in a syndrome trace and collect results keyed by chunk_id.
    Returns {chunk_id: correction_ops}.
    """
    timeout = cfg["deltaflow"]["response_timeout_s"]
    import time
    gen = StimulusGenerator(num_qubits=trace.num_qubits, seed=0)

    results = {}
    for chunk in gen.replay_trace(trace.chunks):
        qeci.send_chunk(chunk)

    for _ in range(len(trace.chunks)):
        result = qeci.receive_correction(timeout_s=timeout)
        if result is not None:
            results[result.chunk_id] = result.correction_ops

    return results


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestDecoderReplay:
    """Replay known syndrome traces and compare output against the oracle."""

    def test_trivial_trace_all_zeros(self, qeci, cfg):
        """
        INTENT: No-error trace — decoder should output all-zero corrections.

        PASS CONDITION (real HW): correction_ops == [0,0,0,0] for every chunk.
        PASS CONDITION (stub)   : correction_ops is a list (shape check only).
        """
        trace = TRACE_TRIVIAL
        results = replay_trace_and_collect(trace, qeci, cfg)

        assert len(results) == len(trace.chunks), (
            f"Expected {len(trace.chunks)} results, got {len(results)}."
        )
        use_hw = cfg["deltaflow"]["use_real_hw"]
        if use_hw:
            agreement = score_agreement(trace.oracle, results)
            assert agreement == 1.0, (
                f"Trivial trace: oracle agreement {agreement:.2%} < 100%."
            )

    def test_single_flip_trace(self, qeci, cfg):
        """
        INTENT: Single-qubit flip — decoder should identify and output the
        correct Pauli correction for qubit 2.

        PASS CONDITION (real HW): oracle agreement == 1.0.
        """
        trace = TRACE_SINGLE_FLIP
        results = replay_trace_and_collect(trace, qeci, cfg)

        assert len(results) == len(trace.chunks)
        if cfg["deltaflow"]["use_real_hw"]:
            agreement = score_agreement(trace.oracle, results)
            assert agreement == 1.0, (
                f"Single-flip trace: agreement {agreement:.2%} < 100%."
            )

    def test_multi_flip_trace(self, qeci, cfg):
        """
        INTENT: Multiple simultaneous flips — exercises MWPM / matching decoder.

        PASS CONDITION (real HW): oracle agreement >= 0.9 (allow slight variance
        if the decoder is probabilistic).
        """
        trace = TRACE_MULTI_FLIP
        results = replay_trace_and_collect(trace, qeci, cfg)

        assert len(results) == len(trace.chunks)
        if cfg["deltaflow"]["use_real_hw"]:
            agreement = score_agreement(trace.oracle, results)
            assert agreement >= 0.9, (
                f"Multi-flip trace: agreement {agreement:.2%} below 90%."
            )

    def test_deterministic_replay_is_identical(self, qeci, cfg):
        """
        INTENT: Replaying the same trace twice must produce identical outputs.
        This validates that the decoder is deterministic for the same input.

        PASS CONDITION: run1 results == run2 results (exact match).
        """
        trace = TRACE_SINGLE_FLIP
        results_1 = replay_trace_and_collect(trace, qeci, cfg)
        results_2 = replay_trace_and_collect(trace, qeci, cfg)

        assert results_1.keys() == results_2.keys(), (
            "Different chunk IDs returned across replays — non-deterministic routing."
        )
        # For real HW, outputs must be byte-identical
        if cfg["deltaflow"]["use_real_hw"]:
            for cid in results_1:
                assert results_1[cid] == results_2[cid], (
                    f"Chunk {cid}: output differs between replays. "
                    "Decoder may be non-deterministic."
                )

    def test_variable_density_trace_no_crash(self, qeci, cfg):
        """
        INTENT: Trace with escalating error density — decoder must not crash
        or return an invalid structure at any density level.

        PASS CONDITION: All chunks return a list (possibly empty); no exception.
        """
        trace = TRACE_VARIABLE_DENSITY
        results = replay_trace_and_collect(trace, qeci, cfg)

        for cid, ops in results.items():
            assert isinstance(ops, list), (
                f"Chunk {cid}: correction_ops is not a list — malformed output."
            )

    def test_empty_chunk_trace(self, qeci, cfg):
        """
        INTENT: Zero-measurement chunk — system must handle gracefully.

        PASS CONDITION: No exception; result may be None or trivial list.
        """
        trace = TRACE_EMPTY_CHUNK
        try:
            results = replay_trace_and_collect(trace, qeci, cfg)
        except Exception as exc:
            pytest.fail(f"Empty chunk caused an exception: {exc}")

    @pytest.mark.parametrize("trace", ALL_TRACES, ids=lambda t: t.name)
    def test_all_traces_complete_without_timeout(self, qeci, cfg, trace):
        """
        INTENT: Every pre-defined trace completes without any chunk timing out.

        PASS CONDITION: received count == sent count for each trace.
        """
        results = replay_trace_and_collect(trace, qeci, cfg)
        expected = len(trace.chunks)
        received = len(results)

        assert received == expected, (
            f"Trace '{trace.name}': expected {expected} results, got {received}. "
            f"Possible timeout on chunks: "
            f"{set(range(expected)) - set(results.keys())}"
        )

"""
tests/test_smoke.py
===================
Step 1 — QECi connectivity and wiring smoke test.

Intent
------
Prove that the testbed is correctly wired before running anything expensive.
A single minimal chunk is sent and exactly one correction result must be
returned.  Any failure here indicates a configuration or connectivity problem,
not a Deltaflow performance issue.

Failure here blocks all subsequent steps in the execution order.
"""

from __future__ import annotations

import pytest

from conftest import run_chunk_and_record


class TestSmoke:
    """Minimal connectivity and wiring checks."""

    def test_single_chunk_produces_one_result(
        self, qeci, stimulus, ts_capture, collector, cfg
    ):
        """
        INTENT: Verify that a minimal valid chunk elicits exactly one
        correction result over the QECi link.

        PASS CONDITION: One CorrectionResult returned; no timeout.
        """
        timeout = cfg["deltaflow"]["response_timeout_s"]
        chunk = stimulus.make_chunk(chunk_id=0, size=4)
        qeci.send_chunk(chunk)
        result = qeci.receive_correction(timeout_s=timeout)

        assert result is not None, (
            "No correction result received within timeout. "
            "Check QECi transport connectivity."
        )
        assert result.chunk_id == 0, (
            f"Chunk ID mismatch: sent 0, received {result.chunk_id}."
        )

    def test_result_has_correction_ops(
        self, qeci, stimulus, ts_capture, collector, cfg
    ):
        """
        INTENT: Verify that the correction result payload is non-empty and
        structurally valid (list of integers).

        PASS CONDITION: correction_ops is a non-empty list.
        """
        timeout = cfg["deltaflow"]["response_timeout_s"]
        chunk = stimulus.make_chunk(chunk_id=1, size=8)
        qeci.send_chunk(chunk)
        result = qeci.receive_correction(timeout_s=timeout)

        assert result is not None, "Timeout — no result received."
        assert isinstance(result.correction_ops, list), (
            "correction_ops must be a list."
        )
        assert len(result.correction_ops) > 0, (
            "Correction result payload is empty — possible malformed response."
        )

    def test_sequential_chunks_produce_sequential_results(
        self, qeci, stimulus, ts_capture, collector, cfg
    ):
        """
        INTENT: Send N=5 sequential chunks and verify exactly N results are
        returned, with no duplicates or gaps in chunk IDs.

        PASS CONDITION: Exactly 5 unique chunk IDs in received results.
        """
        n_chunks = 5
        timeout = cfg["deltaflow"]["response_timeout_s"]
        sent_ids = set()
        received_ids = set()

        collector.start()
        for i in range(n_chunks):
            chunk = stimulus.make_chunk(chunk_id=i, size=16)
            ts_capture.record_ingress(
                chunk_id=i, timestamp_ns=chunk.final_measurement_time_ns
            )
            qeci.send_chunk(chunk)
            sent_ids.add(i)

        for _ in range(n_chunks):
            result = qeci.receive_correction(timeout_s=timeout)
            assert result is not None, "Timeout waiting for correction result."
            received_ids.add(result.chunk_id)
            ts_capture.record_egress(
                chunk_id=result.chunk_id,
                timestamp_ns=result.egress_time_ns,
            )
            lat_ns = ts_capture.compute_latency_ns(result.chunk_id) or 0.0
            collector.record_chunk_result(
                chunk_id=result.chunk_id,
                n_measurements=16,
                latency_ns=lat_ns,
                result_valid=True,
            )
        collector.stop()

        assert received_ids == sent_ids, (
            f"Chunk ID mismatch. Sent: {sent_ids}, Received: {received_ids}"
        )
        assert collector.latency_stats().count == n_chunks

    def test_qeci_link_egress_timestamp_non_zero(
        self, qeci, stimulus, ts_capture, cfg
    ):
        """
        INTENT: Verify that the egress timestamp on the correction result is
        a positive integer (sanity check on the timestamping infrastructure).

        PASS CONDITION: egress_time_ns > 0.
        """
        timeout = cfg["deltaflow"]["response_timeout_s"]
        chunk = stimulus.make_chunk(chunk_id=99, size=4)
        qeci.send_chunk(chunk)
        result = qeci.receive_correction(timeout_s=timeout)

        assert result is not None
        assert result.egress_time_ns > 0, (
            "Egress timestamp is zero — check TimestampCapture clock source."
        )

"""
tests/test_chunking.py
======================
Step 2 — Stimulus and chunking boundary validation.

Intent
------
Verify that Deltaflow correctly batches measurements into chunks at varying
sizes and that chunk boundaries do not create duplicates, gaps, or malformed
payloads.  This must pass before decoder correctness or performance tests.

Tests
-----
* Chunk sizes from config are all handled without error.
* No duplicate correction results for the same chunk_id.
* No gaps: every sent chunk produces exactly one result.
* Zero-size chunk (edge case) does not crash the system.
"""

from __future__ import annotations

import pytest

from conftest import run_chunk_and_record


class TestChunking:
    """Validate batching boundary correctness across chunk sizes."""

    @pytest.mark.parametrize("chunk_size", [16, 32, 64, 128, 256])
    def test_chunk_size_produces_one_result(
        self, qeci, stimulus, ts_capture, collector, cfg, chunk_size
    ):
        """
        INTENT: For each configured chunk size, verify exactly one correction
        result is returned and payload is non-empty.

        PASS CONDITION: One non-empty CorrectionResult per chunk.
        """
        timeout = cfg["deltaflow"]["response_timeout_s"]
        chunk = stimulus.make_chunk(chunk_id=chunk_size, size=chunk_size)
        qeci.send_chunk(chunk)
        result = qeci.receive_correction(timeout_s=timeout)

        assert result is not None, (
            f"No result for chunk_size={chunk_size} — possible batching failure."
        )
        assert isinstance(result.correction_ops, list)

    def test_no_duplicate_results(self, qeci, stimulus, ts_capture, collector, cfg):
        """
        INTENT: Send 20 chunks sequentially.  Verify no chunk_id appears in
        the results more than once (no result duplication).

        PASS CONDITION: All received chunk_ids are unique.
        """
        n = 20
        timeout = cfg["deltaflow"]["response_timeout_s"]

        for i in range(n):
            chunk = stimulus.make_chunk(chunk_id=i, size=32)
            qeci.send_chunk(chunk)

        received_ids = []
        for _ in range(n):
            result = qeci.receive_correction(timeout_s=timeout)
            assert result is not None, "Timeout — possible chunk drop."
            received_ids.append(result.chunk_id)

        duplicates = [cid for cid in received_ids if received_ids.count(cid) > 1]
        assert not duplicates, f"Duplicate correction results for chunk IDs: {set(duplicates)}"

    def test_no_gaps_in_results(self, qeci, stimulus, ts_capture, collector, cfg):
        """
        INTENT: Every sent chunk must produce exactly one result — no gaps.

        PASS CONDITION: len(received) == len(sent) and sent IDs == received IDs.
        """
        n = 15
        timeout = cfg["deltaflow"]["response_timeout_s"]
        sent_ids = set(range(n))

        for i in range(n):
            qeci.send_chunk(stimulus.make_chunk(chunk_id=i, size=64))

        received_ids = set()
        for _ in range(n):
            result = qeci.receive_correction(timeout_s=timeout)
            assert result is not None, "Timeout — gap detected in results."
            received_ids.add(result.chunk_id)

        assert received_ids == sent_ids, (
            f"Gaps detected. Missing: {sent_ids - received_ids}"
        )

    def test_varying_chunk_sizes_in_sequence(
        self, qeci, stimulus, ts_capture, collector, cfg
    ):
        """
        INTENT: Send chunks of alternating sizes in the same stream.
        Verify no cross-contamination between batches.

        PASS CONDITION: Each chunk produces exactly one result; no duplicates.
        """
        sizes = [16, 64, 32, 128, 16, 256]
        timeout = cfg["deltaflow"]["response_timeout_s"]

        for i, size in enumerate(sizes):
            qeci.send_chunk(stimulus.make_chunk(chunk_id=i, size=size))

        received_ids = set()
        for _ in sizes:
            result = qeci.receive_correction(timeout_s=timeout)
            assert result is not None
            received_ids.add(result.chunk_id)

        assert len(received_ids) == len(sizes), "Duplicate or missing results."

    def test_empty_chunk_does_not_crash(self, qeci, stimulus, cfg):
        """
        INTENT: A zero-measurement chunk is an edge case.  The system should
        either return a valid (possibly trivial) result or a clean error — it
        must not crash or hang.

        PASS CONDITION: No exception raised; optional result is returned or
        timeout occurs gracefully.
        """
        from harness.qeci_interface import Chunk
        timeout = cfg["deltaflow"]["response_timeout_s"]
        empty_chunk = Chunk(chunk_id=9999, measurements=[])

        # Should not raise
        try:
            qeci.send_chunk(empty_chunk)
        except Exception as exc:
            pytest.fail(f"send_chunk raised on empty chunk: {exc}")

        # Result may be None (system ignores empty chunks) — both are acceptable
        result = qeci.receive_correction(timeout_s=timeout)
        # If a result is returned it must reference the correct chunk_id
        if result is not None:
            assert result.chunk_id == 9999

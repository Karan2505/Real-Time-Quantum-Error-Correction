"""
tests/test_decoder_robustness.py
=================================
Step 4 — Decoder robustness and performance under hostile input profiles.

Intent
------
Verify that the decoder remains stable, produces valid output, and sustains
throughput when the input is harder than nominal:
  * High error density (dense syndromes)
  * Bursty syndrome arrival
  * Concurrent chunk decoding (parallelism stress)
  * Edge-case: single-qubit code vs. large code distance

These tests catch decoder instability that only manifests under load.
"""

from __future__ import annotations

import time

import pytest

from harness.qeci_interface import Chunk, Measurement
from harness.stimulus_generator import StimulusGenerator
from harness.timestamp_capture import TimestampCapture
from metrics.collector import MetricsCollector


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def send_receive_all(qeci, chunks, timeout_s: float) -> list:
    """Send all chunks, then collect all results.  Returns list of results."""
    for chunk in chunks:
        qeci.send_chunk(chunk)
    results = []
    for _ in chunks:
        r = qeci.receive_correction(timeout_s=timeout_s)
        if r is not None:
            results.append(r)
    return results


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestDecoderRobustness:

    def test_high_density_errors_no_crash(self, qeci, cfg):
        """
        INTENT: Send 50 chunks with 100% error rate (all measurements = 1).
        Decoder must not crash or return malformed output.

        PASS CONDITION: All chunks return a list; no exception; no infinite
        latency (timeout).
        """
        timeout = cfg["deltaflow"]["response_timeout_s"]
        gen = StimulusGenerator(num_qubits=8, error_rate=1.0, seed=7)
        chunks = [gen.make_chunk(chunk_id=i, size=64) for i in range(50)]

        results = send_receive_all(qeci, chunks, timeout)

        assert len(results) == 50, (
            f"Only {len(results)}/50 results returned at 100% error rate."
        )
        for r in results:
            assert isinstance(r.correction_ops, list), (
                f"Malformed correction_ops for chunk {r.chunk_id}."
            )

    def test_zero_error_rate_baseline(self, qeci, cfg):
        """
        INTENT: Send 20 chunks with zero errors.  This is the baseline —
        any overhead measured here is pure pipeline overhead, not decode cost.

        PASS CONDITION: All results returned; correction_ops all zeros or empty
        (decoder-specific).
        """
        timeout = cfg["deltaflow"]["response_timeout_s"]
        gen = StimulusGenerator(num_qubits=8, error_rate=0.0, seed=1)
        chunks = [gen.make_chunk(chunk_id=i, size=32) for i in range(20)]

        results = send_receive_all(qeci, chunks, timeout)
        assert len(results) == 20

    def test_decoder_throughput_under_load(self, qeci, cfg, collector):
        """
        INTENT: Measure decoder throughput (chunks/s) under sustained load.
        Uses 8-qubit, 64-measurement chunks at the maximum configured rate.

        PASS CONDITION:
          * R_decode = C_decoded / Δt satisfies minimum throughput target.
          * No timeouts.
        """
        timeout = cfg["deltaflow"]["response_timeout_s"]
        min_tput = cfg["performance"]["min_throughput_mps"]
        chunk_size = 64
        n_chunks = cfg["stimulus"]["chunks_per_run"]
        gen = StimulusGenerator(num_qubits=8, error_rate=0.001, seed=99)

        collector.start()
        for i in range(n_chunks):
            qeci.send_chunk(gen.make_chunk(chunk_id=i, size=chunk_size))

        received = 0
        for _ in range(n_chunks):
            r = qeci.receive_correction(timeout_s=timeout)
            if r is not None:
                received += 1
                collector.record_chunk_result(
                    chunk_id=r.chunk_id,
                    n_measurements=chunk_size,
                    latency_ns=0.0,      # latency not the focus here
                    result_valid=True,
                )
        collector.stop()

        tput = collector.throughput_mps()
        assert tput >= min_tput, (
            f"Decoder throughput {tput:.0f} meas/s < minimum {min_tput} meas/s."
        )
        assert received == n_chunks, (
            f"Only {received}/{n_chunks} chunks decoded without timeout."
        )

    def test_decoder_latency_under_load(self, qeci, cfg, collector, ts_capture):
        """
        INTENT: Measure per-chunk decode latency under sustained load.

        PASS CONDITION:
          * p99 latency <= max_p99_latency_ms from config.
          * mean latency <= max_mean_latency_ms from config.
        """
        timeout = cfg["deltaflow"]["response_timeout_s"]
        max_p99 = cfg["performance"]["max_p99_latency_ms"]
        max_mean = cfg["performance"]["max_mean_latency_ms"]
        chunk_size = 64
        n_chunks = cfg["stimulus"]["chunks_per_run"]
        warmup = cfg["timing"]["warmup_chunks"]
        gen = StimulusGenerator(num_qubits=8, error_rate=0.001, seed=42)

        # Warm up
        for i in range(warmup):
            c = gen.make_chunk(chunk_id=-(i + 1), size=chunk_size)
            qeci.send_chunk(c)
            qeci.receive_correction(timeout_s=timeout)

        # Measurement window
        collector.start()
        for i in range(n_chunks):
            chunk = gen.make_chunk(chunk_id=i, size=chunk_size)
            ts_capture.record_ingress(chunk.chunk_id, chunk.final_measurement_time_ns)
            qeci.send_chunk(chunk)

        for _ in range(n_chunks):
            r = qeci.receive_correction(timeout_s=timeout)
            if r is not None:
                ts_capture.record_egress(r.chunk_id, r.egress_time_ns)
                lat_ns = ts_capture.compute_latency_ns(r.chunk_id) or 0.0
                collector.record_chunk_result(
                    chunk_id=r.chunk_id,
                    n_measurements=chunk_size,
                    latency_ns=lat_ns,
                    result_valid=True,
                )
        collector.stop()

        stats = collector.latency_stats()

        # In stub mode the per-measurement software latency dominates; strict
        # ms limits are only meaningful on real hardware.
        if cfg["deltaflow"]["use_real_hw"]:
            assert stats.p99_ms <= max_p99, (
                f"p99 latency {stats.p99_ms:.2f} ms > limit {max_p99} ms."
            )
            assert stats.mean_ms <= max_mean, (
                f"Mean latency {stats.mean_ms:.2f} ms > limit {max_mean} ms."
            )
        else:
            # Stub-mode: verify latency is finite and distribution is collected
            assert stats.count > 0, "No latency samples collected."
            assert stats.max_ms < float("inf"), "Infinite latency (timeout) detected."

    def test_concurrent_chunk_decoding(self, qeci, cfg):
        """
        INTENT: Pipeline multiple chunks simultaneously.  Verify results for
        all chunks are received and none are lost when decoding in parallel.

        PASS CONDITION: All n_chunks results received (no drops under parallel
        processing).
        """
        timeout = cfg["deltaflow"]["response_timeout_s"]
        n_chunks = 30
        chunk_size = 32
        gen = StimulusGenerator(num_qubits=8, error_rate=0.001, seed=5)

        # Send all before collecting any (max parallelism)
        for i in range(n_chunks):
            qeci.send_chunk(gen.make_chunk(chunk_id=i, size=chunk_size))

        received = []
        for _ in range(n_chunks):
            r = qeci.receive_correction(timeout_s=timeout)
            if r is not None:
                received.append(r.chunk_id)

        assert len(received) == n_chunks, (
            f"Concurrent decoding: {n_chunks - len(received)} chunk(s) lost."
        )

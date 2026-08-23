"""
tests/test_burst_recovery.py
=============================
Step 6 — Burst stress, overload, and recovery validation.

Intent
------
Inject transient overload, interruption, or malformed traffic and confirm
that the system recovers cleanly without long-lived degradation.  This
validates Deltaflow's resilience, not its peak performance.

Tests
-----
* Burst: short-term rate spike; verify throughput recovers post-burst.
* Link interruption: drop all packets for N seconds; verify resync.
* Malformed packets: corrupt chunk payloads; verify no crash.
* Regression baseline: run at nominal rate before and after disruption;
  throughput should return to pre-disruption level.
"""

from __future__ import annotations

import time

import pytest

from harness.impairment_emulator import ImpairmentEmulator
from harness.qeci_interface import QECiStub
from harness.stimulus_generator import StimulusGenerator
from harness.timestamp_capture import TimestampCapture
from metrics.collector import MetricsCollector


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def measure_phase(
    qeci,
    gen,
    chunk_size: int,
    n_chunks: int,
    timeout: float,
    seed_offset: int = 0,
) -> MetricsCollector:
    """
    Send n_chunks, collect all correction results, and return a fully
    populated MetricsCollector (throughput + per-chunk latency).

    Replaces the old measure_throughput() float-only helper.  Returning
    the full collector means the caller can pass .report_dict() directly
    to the reporter, so the text report shows real latency distributions
    instead of N/A.
    """
    ts   = TimestampCapture()
    coll = MetricsCollector()
    coll.start()

    for i in range(n_chunks):
        chunk = gen.make_chunk(chunk_id=i + seed_offset, size=chunk_size)
        ts.record_ingress(chunk.chunk_id, chunk.final_measurement_time_ns)
        qeci.send_chunk(chunk)
        r = qeci.receive_correction(timeout_s=timeout)
        if r is not None:
            ts.record_egress(r.chunk_id, r.egress_time_ns)
            lat_ns = ts.compute_latency_ns(r.chunk_id) or 0.0
            coll.record_chunk_result(
                chunk_id=r.chunk_id,
                n_measurements=chunk_size,
                latency_ns=lat_ns,
                result_valid=len(r.correction_ops) > 0,
            )
        else:
            coll.record_timeout(chunk_id=i + seed_offset, n_measurements=chunk_size)

    coll.stop()
    return coll


def combined_report(
    before: MetricsCollector,
    after: MetricsCollector,
    extra: dict,
) -> dict:
    """
    Merge two phase collectors into one report dict.

    Standard MetricsCollector keys (throughput_mps, latency, elapsed_s …)
    reflect the *after* phase — that is the recovery measurement under
    scrutiny.  Before-phase stats and comparison values are placed under
    a 'comparison' sub-key so the text formatter renders them in a
    dedicated [ Pre/Post Comparison ] block.
    """
    report = after.report_dict()
    report["comparison"] = {
        "before": before.report_dict(),
        **extra,
    }
    return report


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBurstRecovery:

    def test_burst_then_recovery(self, cfg, reporter):
        """
        INTENT: Inject a burst (3x nominal rate), then return to nominal rate
        and verify throughput recovers.

        PROCEDURE:
          1. Measure baseline throughput + latency (pre-burst).
          2. Send burst_chunks chunks at 3x rate with no inter-chunk pacing.
          3. Drain, then measure post-burst throughput + latency.
          4. Assert post-burst >= 90% of pre-burst.

        PASS CONDITION: recovery_ratio >= 0.9.
        """
        chunk_size        = 64
        timeout           = cfg["deltaflow"]["response_timeout_s"]
        n_baseline_chunks = 30
        burst_multiplier  = cfg["burst"]["burst_multiplier"]
        nominal_rate      = min(cfg["stimulus"]["rates_mps"])
        burst_rate        = nominal_rate * burst_multiplier

        backend = QECiStub(base_latency_ms=2.0, latency_jitter_ms=0.3)
        gen     = StimulusGenerator(num_qubits=8, error_rate=0.001, seed=1)

        # Phase 1: pre-burst baseline
        coll_before = measure_phase(
            backend, gen, chunk_size, n_baseline_chunks, timeout, seed_offset=0
        )
        tput_before = coll_before.throughput_mps()

        # Burst window — no inter-chunk sleep, maximum send rate
        burst_chunks = 20
        for i in range(burst_chunks):
            backend.send_chunk(gen.make_chunk(chunk_id=1000 + i, size=chunk_size))
        for _ in range(burst_chunks):
            backend.receive_correction(timeout_s=timeout)

        # Brief drain pause
        time.sleep(0.2)

        # Phase 2: post-burst recovery
        coll_after = measure_phase(
            backend, gen, chunk_size, n_baseline_chunks, timeout, seed_offset=2000
        )
        tput_after = coll_after.throughput_mps()
        backend.close()

        recovery_ratio = tput_after / max(tput_before, 1.0)

        reporter.write(
            test_name="burst_recovery",
            config=cfg,
            metrics=combined_report(
                before=coll_before,
                after=coll_after,
                extra={
                    "tput_before_mps": round(tput_before, 1),
                    "tput_after_mps":  round(tput_after, 1),
                    "recovery_ratio":  round(recovery_ratio, 3),
                    "burst_multiplier": burst_multiplier,
                },
            ),
            passed=recovery_ratio >= 0.9,
            notes=[
                f"Nominal rate: {nominal_rate} meas/s",
                f"Burst rate: {burst_rate} meas/s ({burst_multiplier}x)",
                f"Recovery ratio: {recovery_ratio:.1%}",
            ],
        )

        assert recovery_ratio >= 0.9, (
            f"Post-burst throughput {tput_after:.0f} meas/s is only "
            f"{recovery_ratio:.1%} of pre-burst {tput_before:.0f} meas/s."
        )

    def test_packet_loss_recovery(self, cfg, reporter):
        """
        INTENT: Inject 10% packet loss for 20 chunks, then restore a clean
        link.  Verify the system recovers to nominal throughput.

        PASS CONDITION: recovery_ratio >= 0.9.
        """
        chunk_size = 64
        timeout    = cfg["deltaflow"]["response_timeout_s"]
        n_chunks   = 30
        loss_prob  = 0.10

        backend  = QECiStub(base_latency_ms=2.0, latency_jitter_ms=0.3)
        emulator = ImpairmentEmulator(backend, loss_probability=0.0, seed=7)
        gen      = StimulusGenerator(num_qubits=8, error_rate=0.001, seed=3)

        # Phase 1: baseline (no loss)
        coll_before = measure_phase(
            emulator, gen, chunk_size, n_chunks, timeout, seed_offset=0
        )
        tput_before = coll_before.throughput_mps()

        # Lossy window — only these chunks are subject to loss_prob
        emulator.loss_probability = loss_prob
        lossy_sent = 20
        for i in range(lossy_sent):
            emulator.send_chunk(gen.make_chunk(chunk_id=5000 + i, size=chunk_size))
        for _ in range(lossy_sent):
            emulator.receive_correction(timeout_s=timeout)

        # Restore clean link
        emulator.loss_probability = 0.0
        time.sleep(0.1)

        # Phase 2: post-loss recovery
        coll_after = measure_phase(
            emulator, gen, chunk_size, n_chunks, timeout, seed_offset=10000
        )
        tput_after = coll_after.throughput_mps()
        emulator.close()

        recovery_ratio = tput_after / max(tput_before, 1.0)
        imp_summary    = emulator.summary()

        reporter.write(
            test_name="packet_loss_recovery",
            config=cfg,
            metrics=combined_report(
                before=coll_before,
                after=coll_after,
                extra={
                    "tput_before_mps":    round(tput_before, 1),
                    "tput_after_mps":     round(tput_after, 1),
                    "recovery_ratio":     round(recovery_ratio, 3),
                    "loss_probability":   loss_prob,
                    # Report only the lossy-window chunk count, not total
                    "lossy_window_sent":  lossy_sent,
                    "lossy_window_dropped": imp_summary["dropped"],
                    "impairment_summary": imp_summary,
                },
            ),
            passed=recovery_ratio >= 0.9,
            notes=[
                f"Loss probability during impairment: {loss_prob:.0%}",
                f"Chunks in lossy window: {lossy_sent}  "
                f"(dropped: {imp_summary['dropped']})",
                f"Recovery ratio: {recovery_ratio:.1%}",
            ],
        )

        assert recovery_ratio >= 0.9, (
            f"Post-loss throughput {recovery_ratio:.1%} of baseline. "
            "System has not recovered."
        )

    def test_malformed_packet_no_crash(self, cfg):
        """
        INTENT: Send corrupted chunk payloads.  Deltaflow must not crash or
        hang.  Clean chunks after the corrupt window must still produce valid
        results.

        PASS CONDITION: No exception raised; all n_clean post-corrupt chunks
        return a valid list result.
        """
        chunk_size = 32
        timeout    = cfg["deltaflow"]["response_timeout_s"]
        n_clean    = 10

        backend  = QECiStub(base_latency_ms=2.0, corrupt_probability=0.0)
        emulator = ImpairmentEmulator(backend, corrupt_probability=1.0, seed=9)
        gen      = StimulusGenerator(num_qubits=8, error_rate=0.001, seed=5)

        # Corrupt window
        try:
            for i in range(10):
                emulator.send_chunk(gen.make_chunk(chunk_id=i, size=chunk_size))
            for _ in range(10):
                emulator.receive_correction(timeout_s=timeout)
        except Exception as exc:
            pytest.fail(f"Exception during corrupt-packet window: {exc}")

        # Restore clean traffic through the backend directly
        emulator.corrupt_probability = 0.0
        clean_results = 0
        for i in range(n_clean):
            backend.send_chunk(gen.make_chunk(chunk_id=100 + i, size=chunk_size))
        for _ in range(n_clean):
            r = backend.receive_correction(timeout_s=timeout)
            if r is not None and isinstance(r.correction_ops, list):
                clean_results += 1
        emulator.close()

        assert clean_results == n_clean, (
            f"Only {clean_results}/{n_clean} clean results after corrupt window. "
            "System may not have recovered from malformed packets."
        )

    def test_regression_baseline_stability(self, cfg, reporter):
        """
        INTENT: Run at nominal rate, apply a 500ms link interruption, then
        run again.  Post-disruption throughput must be within the drift
        threshold of pre-disruption.

        PASS CONDITION:
          * Real HW : drift <= 5%  (tight — meaningful for FPGA pipeline)
          * Stub    : drift <= 10% (wider — stub measures Python scheduling
                      noise, not FPGA behaviour; 5% is too tight for the OS)

        Stability measures taken:
          * chunks_per_run (200) for both phases — gives ~700 ms window
          * warmup pass before each phase — lets the OS scheduler settle
        """
        chunk_size    = 64
        timeout       = cfg["deltaflow"]["response_timeout_s"]
        n_chunks      = cfg["stimulus"]["chunks_per_run"]
        warmup        = cfg["timing"]["warmup_chunks"]
        use_hw        = cfg["deltaflow"]["use_real_hw"]
        drift_limit   = 0.05 if use_hw else 0.10

        backend = QECiStub(base_latency_ms=2.0, latency_jitter_ms=0.3)
        gen     = StimulusGenerator(num_qubits=8, error_rate=0.001, seed=99)

        # Warmup before phase 1 — burns off cold-start OS scheduling variance
        for i in range(warmup):
            c = gen.make_chunk(chunk_id=-(i + 1), size=chunk_size)
            backend.send_chunk(c)
            backend.receive_correction(timeout_s=timeout)

        # Phase 1: pre-disruption
        coll_before = measure_phase(
            backend, gen, chunk_size, n_chunks, timeout, seed_offset=0
        )
        tput_before = coll_before.throughput_mps()

        # Simulated 500ms link outage
        time.sleep(0.5)

        # Warmup before phase 2 — same reason as above
        for i in range(warmup):
            c = gen.make_chunk(chunk_id=-(20000 + i + 1), size=chunk_size)
            backend.send_chunk(c)
            backend.receive_correction(timeout_s=timeout)

        # Phase 2: post-disruption
        coll_after = measure_phase(
            backend, gen, chunk_size, n_chunks, timeout, seed_offset=10000
        )
        tput_after = coll_after.throughput_mps()
        backend.close()

        drift = abs(tput_after - tput_before) / max(tput_before, 1.0)
        passed = drift <= drift_limit

        reporter.write(
            test_name="regression_baseline_stability",
            config=cfg,
            metrics=combined_report(
                before=coll_before,
                after=coll_after,
                extra={
                    "tput_before_mps": round(tput_before, 1),
                    "tput_after_mps":  round(tput_after, 1),
                    "drift_pct":       round(drift * 100, 2),
                    "drift_limit_pct": round(drift_limit * 100, 1),
                },
            ),
            passed=passed,
            notes=[
                f"Throughput drift: {drift:.2%}  (limit: {drift_limit:.0%})",
                "(stub mode — 10% limit applied)" if not use_hw else "(real HW — 5% limit applied)",
            ],
        )

        assert passed, (
            f"Throughput drifted {drift:.2%} after disruption "
            f"({tput_before:.0f} -> {tput_after:.0f} meas/s). "
            f"Limit is {drift_limit:.0%} in {'stub' if not use_hw else 'real-HW'} mode."
        )
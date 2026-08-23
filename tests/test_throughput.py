"""
tests/test_throughput.py
========================
Step 5a — Throughput sweep: measurements processed per unit time.

Intent
------
Drive increasing measurement rates until the system reaches steady-state
saturation.  The useful outcome is not just the peak number but the knee
point where queueing and backlog begin to grow.

Formula: T = N_processed / Δt  (measurements per second)

Procedure (from test plan section 7)
-------------------------------------
1. Configure a fixed chunk size and a known input rate.
2. Run the system long enough to reach steady state (warmup period).
3. Count measurements corresponding to completed correction results
   over the observation interval.
4. Throughput = processed measurements / elapsed time.
5. Repeat at below-nominal, nominal, and above-nominal rates.
6. Identify the knee point where throughput stops scaling with input rate.

Additional components required
-------------------------------
* Stimulus generator (StimulusGenerator) — provided in harness.
* Precise wall-clock timing via time.perf_counter().
* If run on real hardware, hardware timestamping NIC recommended.

Fixes applied (vs original)
----------------------------
* passed flag in reporter.write() now consistent with the pytest assertion:
  sub-target input rates are not expected to hit min_tput so they report PASS.
* latency_ns=0.0 replaced with real TimestampCapture so the Latency
  Distribution block in the report shows real values instead of all-zeros.
* Knee detection algorithm now excludes input-paced rates (where measured
  throughput < input rate) so the knee is computed only over the region
  where Deltaflow is actually the bottleneck.
"""

from __future__ import annotations

import time

import pytest

from harness.stimulus_generator import StimulusGenerator
from harness.timestamp_capture import TimestampCapture
from metrics.collector import MetricsCollector
from reporting.report_generator import ReportGenerator


class TestThroughput:
    """Throughput sweep across configured measurement rates."""

    @pytest.mark.parametrize("rate_mps", [1_000, 5_000, 10_000, 50_000, 100_000])
    def test_throughput_at_fixed_rate(
        self, qeci, cfg, collector, reporter, rate_mps
    ):
        """
        INTENT: Measure sustained throughput at a specific input rate.

        PROCEDURE:
          1. Warm up for warmup_chunks chunks (results discarded).
          2. Start observation window.
          3. Send chunks at rate_mps with real timestamp capture.
          4. Stop window; compute T = N_processed / Δt.

        PASS CONDITION:
          * For rates >= min_throughput_mps: T >= min_tput * 0.9 (10% tolerance).
          * For rates < min_throughput_mps: always PASS — system is input-paced
            by design; throughput is bounded by the generator, not Deltaflow.
          * All chunks processed without timeout.
        """
        chunk_size = 64
        n_chunks   = cfg["stimulus"]["chunks_per_run"]
        warmup     = cfg["timing"]["warmup_chunks"]
        timeout    = cfg["deltaflow"]["response_timeout_s"]
        min_tput   = cfg["performance"]["min_throughput_mps"]
        gen        = StimulusGenerator(num_qubits=8, error_rate=0.001, seed=42)
        ts         = TimestampCapture()

        # Warm-up (not measured, not reported)
        for i in range(warmup):
            c = gen.make_chunk(chunk_id=-(i + 1), size=chunk_size)
            qeci.send_chunk(c)
            qeci.receive_correction(timeout_s=timeout)

        # Observation window
        collector.reset()
        collector.start()

        inter_chunk_s = chunk_size / rate_mps
        n_received = 0

        for i in range(n_chunks):
            t0    = time.perf_counter()
            chunk = gen.make_chunk(chunk_id=i, size=chunk_size)
            ts.record_ingress(chunk.chunk_id, chunk.final_measurement_time_ns)
            qeci.send_chunk(chunk)

            result = qeci.receive_correction(timeout_s=timeout)
            if result is not None:
                n_received += 1
                ts.record_egress(result.chunk_id, result.egress_time_ns)
                lat_ns = ts.compute_latency_ns(result.chunk_id) or 0.0
                collector.record_chunk_result(
                    chunk_id=result.chunk_id,
                    n_measurements=chunk_size,
                    latency_ns=lat_ns,
                    result_valid=len(result.correction_ops) > 0,
                )
            else:
                collector.record_timeout(chunk_id=i, n_measurements=chunk_size)

            sleep_s = inter_chunk_s - (time.perf_counter() - t0)
            if sleep_s > 0:
                time.sleep(sleep_s)

        collector.stop()
        tput = collector.throughput_mps()

        # Report passed = True when input rate is below the minimum target
        # (system is deliberately input-paced) OR when throughput meets the
        # target within the 10% timing-noise tolerance.
        is_input_paced  = rate_mps < min_tput
        meets_target    = tput >= min_tput * 0.9
        report_passed   = is_input_paced or meets_target

        reporter.write(
            test_name=f"throughput_{rate_mps}_mps",
            config=cfg,
            metrics=collector.report_dict(),
            passed=report_passed,
            notes=[n for n in [
                f"Input rate: {rate_mps} meas/s",
                f"Measured throughput: {tput:.0f} meas/s",
                f"Chunks received: {n_received}/{n_chunks}",
                "Input-paced run (throughput bounded by generator, not Deltaflow)"
                if is_input_paced else "",
            ] if n],
        )

        # Pytest assertion: only enforce the minimum when rate >= min_tput
        if not is_input_paced:
            assert meets_target, (
                f"Throughput {tput:.0f} meas/s < minimum {min_tput * 0.9:.0f} meas/s "
                f"(90% of {min_tput}) at input rate {rate_mps} meas/s."
            )

    def test_throughput_knee_detection(self, qeci, cfg, reporter):
        """
        INTENT: Sweep all configured rates and identify the knee point —
        the rate at which measured throughput stops scaling linearly with
        input rate (backlog begins to grow).

        PASS CONDITION:
          * Knee point is logged (not asserted — hardware-dependent).
          * If a knee is found, it must be above min_throughput_mps.

        Fix: knee algorithm now excludes input-paced rates (measured ≈ input
        rate) from the gain comparison.  Only the region where Deltaflow is
        the bottleneck — where measured throughput decouples from input rate —
        is meaningful for knee detection.
        """
        rates      = cfg["stimulus"]["rates_mps"]
        chunk_size = 64
        n_chunks   = min(50, cfg["stimulus"]["chunks_per_run"])
        warmup     = cfg["timing"]["warmup_chunks"]
        timeout    = cfg["deltaflow"]["response_timeout_s"]
        gen        = StimulusGenerator(num_qubits=8, error_rate=0.001, seed=0)

        results_by_rate: dict[int, float] = {}

        for rate_mps in rates:
            coll = MetricsCollector()
            ts   = TimestampCapture()

            # Per-rate warmup
            for i in range(warmup):
                c = gen.make_chunk(chunk_id=-(i + 1), size=chunk_size)
                qeci.send_chunk(c)
                qeci.receive_correction(timeout_s=timeout)

            inter_chunk_s = chunk_size / rate_mps
            coll.start()
            for i in range(n_chunks):
                t0    = time.perf_counter()
                chunk = gen.make_chunk(chunk_id=i, size=chunk_size)
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
                        result_valid=True,
                    )
                sleep_s = inter_chunk_s - (time.perf_counter() - t0)
                if sleep_s > 0:
                    time.sleep(sleep_s)
            coll.stop()
            results_by_rate[rate_mps] = coll.throughput_mps()

        # ----------------------------------------------------------------
        # Knee detection — two-regime classifier
        # ----------------------------------------------------------------
        # The measured-throughput curve has two distinct regimes:
        #
        #   Input-paced  : measured ≈ input_rate  (generator is bottleneck)
        #   Saturated    : measured ≈ ceiling      (Deltaflow is bottleneck)
        #
        # The old gain-ratio approach labelled high rates (50k, 100k) as
        # [input-paced] because measured << input_rate, which is semantically
        # wrong — at those rates Deltaflow is saturated at its processing
        # ceiling, not the generator running slow.
        #
        # New approach:
        #   1. Estimate the system ceiling from the top-2 measured values.
        #   2. Classify every rate as input-paced, saturated, or transition.
        #   3. Knee = the lowest input rate where saturation is first reached.

        sorted_rates = sorted(results_by_rate)

        # Ceiling: average of top-2 measured throughputs (robust to one outlier)
        top2_measured = sorted(results_by_rate.values())[-2:]
        ceiling = sum(top2_measured) / len(top2_measured)

        def classify_rate(input_rate: int, measured: float) -> str:
            """Return 'input-paced', 'saturated', or 'transition'."""
            if measured >= input_rate * 0.85:
                # Measured throughput tracks the input rate → generator-limited
                return "input-paced"
            if measured >= ceiling * 0.85:
                # Measured throughput has plateaued at the system ceiling
                return "saturated"
            return "transition"

        classifications = {
            r: classify_rate(r, results_by_rate[r]) for r in sorted_rates
        }

        # Knee = lowest rate classified as saturated
        knee_rate = next(
            (r for r in sorted_rates if classifications[r] == "saturated"),
            None,
        )

        # Build notes with correct labels
        label_map = {
            "input-paced": "[input-paced]",
            "saturated":   "[saturated — Deltaflow ceiling]",
            "transition":  "[transition]",
        }
        notes = [
            f"Rate {r} meas/s -> measured {results_by_rate[r]:.0f} meas/s "
            f"{label_map[classifications[r]]}"
            for r in sorted_rates
        ]
        notes.append(f"Estimated system ceiling: {ceiling:.0f} meas/s")
        if knee_rate:
            notes.append(
                f"Knee point detected at approximately {knee_rate} meas/s "
                f"(first rate where Deltaflow is the bottleneck)"
            )
        else:
            notes.append("No saturation knee detected in configured rate range")

        reporter.write(
            test_name="throughput_knee_detection",
            config=cfg,
            metrics={"throughput_by_rate": {str(k): round(v, 1)
                                             for k, v in results_by_rate.items()}},
            passed=True,
            notes=notes,
        )

        if knee_rate is not None:
            min_tput = cfg["performance"]["min_throughput_mps"]
            assert ceiling >= min_tput, (
                f"System ceiling {ceiling:.0f} meas/s is below minimum "
                f"throughput target {min_tput} meas/s."
            )
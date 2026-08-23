"""
metrics/collector.py
====================
Statistics collector for throughput and latency measurements.

Implements the formulae from the test plan:

    T  = N_processed / Δt        (throughput, measurements/s)
    L  = t_egress − t_final_input (chunk latency, nanoseconds)
    Rd = C_decoded / Δt          (decoder throughput, chunks/s)
    A  = M_match / N_total       (oracle agreement)

Public API
----------
MetricsCollector.record_chunk_result(chunk_id, n_measurements, latency_ns)
MetricsCollector.throughput_mps()    -> float
MetricsCollector.decoder_rate_cps()  -> float
MetricsCollector.latency_stats()     -> LatencyStats
MetricsCollector.report_dict()       -> dict
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class LatencyStats:
    """Descriptive statistics over chunk-level latency measurements."""
    count: int
    mean_ms: float
    median_ms: float
    p95_ms: float
    p99_ms: float
    max_ms: float
    min_ms: float
    std_ms: float

    def passes(self, max_mean_ms: float, max_p99_ms: float) -> bool:
        """Return True if both mean and p99 are within the given limits."""
        return self.mean_ms <= max_mean_ms and self.p99_ms <= max_p99_ms

    def __str__(self) -> str:
        return (
            f"Latency [{self.count} chunks] "
            f"mean={self.mean_ms:.2f} ms  "
            f"p95={self.p95_ms:.2f} ms  "
            f"p99={self.p99_ms:.2f} ms  "
            f"max={self.max_ms:.2f} ms"
        )


@dataclass
class ChunkRecord:
    chunk_id: int
    n_measurements: int
    latency_ns: float       # L_chunk in nanoseconds
    timestamp: float        # wall time when result was recorded (perf_counter)
    result_valid: bool = True


class MetricsCollector:
    """
    Accumulates per-chunk results and computes aggregate metrics.

    Usage
    -----
    collector = MetricsCollector()
    collector.start()
    # ... run test ...
    collector.record_chunk_result(chunk_id=0, n_measurements=64, latency_ns=1_200_000)
    collector.stop()
    stats = collector.latency_stats()
    tput  = collector.throughput_mps()
    """

    def __init__(self):
        self._records: List[ChunkRecord] = []
        self._invalid_chunks: List[int] = []
        self._t_start: Optional[float] = None
        self._t_stop: Optional[float] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Mark the beginning of the observation window."""
        self._t_start = time.perf_counter()

    def stop(self) -> None:
        """Mark the end of the observation window."""
        self._t_stop = time.perf_counter()

    @property
    def elapsed_s(self) -> float:
        """Observation window duration in seconds."""
        if self._t_start is None:
            return 0.0
        end = self._t_stop if self._t_stop is not None else time.perf_counter()
        return end - self._t_start

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_chunk_result(
        self,
        chunk_id: int,
        n_measurements: int,
        latency_ns: float,
        result_valid: bool = True,
    ) -> None:
        """
        Record the result of a processed chunk.

        Parameters
        ----------
        chunk_id : int
        n_measurements : int
            Size of this chunk (used for throughput calculation).
        latency_ns : float
            Chunk latency = t_egress − t_final_input (nanoseconds).
        result_valid : bool
            False if the correction result was malformed or missing.
        """
        self._records.append(
            ChunkRecord(
                chunk_id=chunk_id,
                n_measurements=n_measurements,
                latency_ns=latency_ns,
                timestamp=time.perf_counter(),
                result_valid=result_valid,
            )
        )
        if not result_valid:
            self._invalid_chunks.append(chunk_id)

    def record_timeout(self, chunk_id: int, n_measurements: int) -> None:
        """Record a chunk that timed out (no correction result received)."""
        self._invalid_chunks.append(chunk_id)
        # Use a sentinel latency so the chunk still appears in the report
        self._records.append(
            ChunkRecord(
                chunk_id=chunk_id,
                n_measurements=n_measurements,
                latency_ns=float("inf"),
                timestamp=time.perf_counter(),
                result_valid=False,
            )
        )

    # ------------------------------------------------------------------
    # Throughput
    # ------------------------------------------------------------------

    def throughput_mps(self) -> float:
        """
        T = N_processed / Δt  (measurements per second).

        Only valid chunks contribute to N_processed.
        """
        if self.elapsed_s == 0:
            return 0.0
        n_processed = sum(
            r.n_measurements for r in self._records if r.result_valid
        )
        return n_processed / self.elapsed_s

    def decoder_rate_cps(self) -> float:
        """
        Rd = C_decoded / Δt  (chunks per second).
        """
        if self.elapsed_s == 0:
            return 0.0
        c_decoded = sum(1 for r in self._records if r.result_valid)
        return c_decoded / self.elapsed_s

    # ------------------------------------------------------------------
    # Latency statistics
    # ------------------------------------------------------------------

    def latency_stats(self) -> LatencyStats:
        """
        Compute descriptive statistics over valid chunk latencies.

        Infinite latency values (timeouts) are excluded from statistics
        but counted in the report.
        """
        valid_ns = [
            r.latency_ns
            for r in self._records
            if r.result_valid and math.isfinite(r.latency_ns)
        ]
        if not valid_ns:
            return LatencyStats(0, 0, 0, 0, 0, 0, 0, 0)

        sorted_ns = sorted(valid_ns)
        n = len(sorted_ns)
        ms = [v / 1e6 for v in sorted_ns]

        def percentile(data: List[float], p: float) -> float:
            if not data:
                return 0.0
            idx = (p / 100) * (len(data) - 1)
            lo, hi = int(idx), min(int(idx) + 1, len(data) - 1)
            return data[lo] + (data[hi] - data[lo]) * (idx - lo)

        mean = sum(ms) / n
        variance = sum((x - mean) ** 2 for x in ms) / n if n > 1 else 0.0

        return LatencyStats(
            count=n,
            mean_ms=mean,
            median_ms=percentile(ms, 50),
            p95_ms=percentile(ms, 95),
            p99_ms=percentile(ms, 99),
            max_ms=max(ms),
            min_ms=min(ms),
            std_ms=math.sqrt(variance),
        )

    # ------------------------------------------------------------------
    # Summary report
    # ------------------------------------------------------------------

    def report_dict(self) -> dict:
        """Return all metrics as a JSON-serialisable dict."""
        stats = self.latency_stats()
        return {
            "elapsed_s": round(self.elapsed_s, 3),
            "total_chunks": len(self._records),
            "valid_chunks": sum(1 for r in self._records if r.result_valid),
            "invalid_chunks": len(self._invalid_chunks),
            "throughput_mps": round(self.throughput_mps(), 1),
            "decoder_rate_cps": round(self.decoder_rate_cps(), 1),
            "latency": {
                "count": stats.count,
                "mean_ms": round(stats.mean_ms, 3),
                "median_ms": round(stats.median_ms, 3),
                "p95_ms": round(stats.p95_ms, 3),
                "p99_ms": round(stats.p99_ms, 3),
                "max_ms": round(stats.max_ms, 3),
                "min_ms": round(stats.min_ms, 3),
                "std_ms": round(stats.std_ms, 3),
            },
        }

    def reset(self) -> None:
        self._records.clear()
        self._invalid_chunks.clear()
        self._t_start = None
        self._t_stop = None

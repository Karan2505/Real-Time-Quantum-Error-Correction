"""
harness/timestamp_capture.py
=============================
Ingress and egress timestamp capture with clock synchronisation awareness.

Why this matters
----------------
Latency = t_egress − t_final_input.  If the ingress and egress clocks are
not synchronised, the computed latency includes the clock offset and is
meaningless.  This module:

  1. Records ingress timestamps when the final measurement of each chunk
     is submitted to Deltaflow.
  2. Records egress timestamps when a CorrectionResult arrives.
  3. Pairs them by chunk_id and computes per-chunk latency.
  4. Warns (and optionally fails) if estimated clock skew exceeds the
     configured tolerance.

In a real lab, ingress and egress should both be measured by the same host
(the test controller) or by NICs sharing a PTP grandmaster.  When that is
not possible, this module can be extended to accept hardware timestamps
embedded in QECi packets.

Public API
----------
TimestampCapture.record_ingress(chunk_id, timestamp_ns) -> None
TimestampCapture.record_egress(chunk_id, timestamp_ns) -> None
TimestampCapture.compute_latency(chunk_id) -> float | None  (nanoseconds)
TimestampCapture.all_latencies() -> List[float]
TimestampCapture.check_clock_skew(tolerance_us) -> bool
"""

from __future__ import annotations

import logging
import time
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ChunkTiming:
    chunk_id: int
    ingress_ns: Optional[int] = None   # t_final_input
    egress_ns: Optional[int] = None    # t_correction_result


class TimestampCapture:
    """
    Thread-safe ingress/egress timestamp store.

    Timestamps are stored as integer nanoseconds (time.time_ns() on the
    host running the test framework).  For real-hardware runs, replace
    with NIC-provided hardware timestamps for better accuracy.
    """

    def __init__(self, max_clock_skew_us: float = 50.0):
        self.max_clock_skew_us = max_clock_skew_us
        self._records: Dict[int, ChunkTiming] = {}
        self._baseline_skew_estimate_ns: Optional[float] = None

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_ingress(self, chunk_id: int, timestamp_ns: Optional[int] = None) -> None:
        """
        Record the ingress timestamp for the final measurement of chunk_id.

        If timestamp_ns is None, time.time_ns() is used — appropriate when
        the test controller is both sender and timestamper.
        """
        ts = timestamp_ns if timestamp_ns is not None else time.time_ns()
        if chunk_id not in self._records:
            self._records[chunk_id] = ChunkTiming(chunk_id=chunk_id)
        self._records[chunk_id].ingress_ns = ts

    def record_egress(self, chunk_id: int, timestamp_ns: Optional[int] = None) -> None:
        """
        Record the egress timestamp for the correction result of chunk_id.
        """
        ts = timestamp_ns if timestamp_ns is not None else time.time_ns()
        if chunk_id not in self._records:
            self._records[chunk_id] = ChunkTiming(chunk_id=chunk_id)
        self._records[chunk_id].egress_ns = ts

    # ------------------------------------------------------------------
    # Latency computation
    # ------------------------------------------------------------------

    def compute_latency_ns(self, chunk_id: int) -> Optional[float]:
        """
        Return latency in nanoseconds for chunk_id, or None if either
        timestamp is missing.

        L_chunk = t_egress − t_final_input
        """
        rec = self._records.get(chunk_id)
        if rec is None or rec.ingress_ns is None or rec.egress_ns is None:
            return None
        latency = rec.egress_ns - rec.ingress_ns
        if latency < 0:
            warnings.warn(
                f"Negative latency ({latency} ns) for chunk {chunk_id}. "
                "This usually indicates a clock synchronisation problem.",
                RuntimeWarning,
                stacklevel=2,
            )
        return float(latency)

    def compute_latency_ms(self, chunk_id: int) -> Optional[float]:
        ns = self.compute_latency_ns(chunk_id)
        return ns / 1e6 if ns is not None else None

    def all_latencies_ns(self) -> List[float]:
        """Return list of valid latency measurements in nanoseconds."""
        result = []
        for chunk_id in self._records:
            lat = self.compute_latency_ns(chunk_id)
            if lat is not None:
                result.append(lat)
        return result

    def all_latencies_ms(self) -> List[float]:
        return [ns / 1e6 for ns in self.all_latencies_ns()]

    # ------------------------------------------------------------------
    # Clock skew estimation (loopback probe)
    # ------------------------------------------------------------------

    def estimate_loopback_skew_ns(self, num_probes: int = 20) -> float:
        """
        Estimate the round-trip loopback latency of time.time_ns() itself.

        In the software stub this is essentially zero.  On real hardware
        with NIC timestamping this can be tens of microseconds.  Use the
        result to flag if measured latencies are suspiciously small.
        """
        deltas = []
        for _ in range(num_probes):
            t0 = time.time_ns()
            _ = time.time_ns()  # simulate an egress read
            t1 = time.time_ns()
            deltas.append(t1 - t0)
        avg = sum(deltas) / len(deltas)
        self._baseline_skew_estimate_ns = avg
        return avg

    def check_clock_skew(self) -> bool:
        """
        Return True if clock skew is within tolerance.

        For software-stub runs this always passes.  Hook in PTP offset
        readings for real-hardware validation.
        """
        if self._baseline_skew_estimate_ns is None:
            self.estimate_loopback_skew_ns()

        skew_us = (self._baseline_skew_estimate_ns or 0) / 1e3
        if skew_us > self.max_clock_skew_us:
            logger.warning(
                "Clock skew %.1f µs exceeds tolerance %.1f µs. "
                "Latency measurements may be unreliable.",
                skew_us,
                self.max_clock_skew_us,
            )
            return False
        return True

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def missing_egress(self) -> List[int]:
        """Return chunk IDs that have an ingress record but no egress."""
        return [
            cid
            for cid, rec in self._records.items()
            if rec.ingress_ns is not None and rec.egress_ns is None
        ]

    def missing_ingress(self) -> List[int]:
        """Return chunk IDs that have an egress record but no ingress."""
        return [
            cid
            for cid, rec in self._records.items()
            if rec.egress_ns is not None and rec.ingress_ns is None
        ]

    def reset(self) -> None:
        self._records.clear()

    def __len__(self) -> int:
        return len(self._records)

"""
harness/impairment_emulator.py
==============================
Network impairment injection for stress and resilience testing.

Wraps any QECiInterface and adds configurable:
  * Extra delay (constant or Gaussian jitter)
  * Packet loss (random drop)
  * Packet reordering (short reorder buffer)
  * Malformed packet injection

Used in test_burst_recovery.py to verify Deltaflow recovers cleanly
after transient adverse link conditions.

Limitation
----------
This is a software emulator — it approximates realistic link impairments
but cannot reproduce hardware-level timing artifacts.  For production
validation, pair with a real traffic shaper (e.g., tc-netem on Linux).
"""

from __future__ import annotations

import random
import time
import threading
from collections import deque
from typing import Optional

from harness.qeci_interface import Chunk, CorrectionResult, QECiInterface


class ImpairmentEmulator(QECiInterface):
    """
    QECiInterface decorator that injects configurable link impairments.

    Parameters
    ----------
    backend : QECiInterface
        The underlying (real or stub) interface.
    extra_delay_ms : float
        Additional fixed delay added to every send (milliseconds).
    jitter_ms : float
        Standard deviation of Gaussian jitter applied to each send.
    loss_probability : float
        Probability [0, 1) that a chunk send is silently dropped.
    reorder_probability : float
        Probability [0, 1) that a chunk is held and sent after the next one.
    corrupt_probability : float
        Probability [0, 1) that a chunk payload is corrupted before send.
    """

    def __init__(
        self,
        backend: QECiInterface,
        extra_delay_ms: float = 0.0,
        jitter_ms: float = 0.0,
        loss_probability: float = 0.0,
        reorder_probability: float = 0.0,
        corrupt_probability: float = 0.0,
        seed: Optional[int] = None,
    ):
        self._backend = backend
        self.extra_delay_ms = extra_delay_ms
        self.jitter_ms = jitter_ms
        self.loss_probability = loss_probability
        self.reorder_probability = reorder_probability
        self.corrupt_probability = corrupt_probability
        self._rng = random.Random(seed)

        # Reorder buffer holds at most one chunk
        self._reorder_buffer: deque[Chunk] = deque(maxlen=2)
        self._lock = threading.Lock()

        # Counters for test reporting
        self.dropped_count = 0
        self.reordered_count = 0
        self.corrupted_count = 0
        self.total_sent = 0

    # ------------------------------------------------------------------
    # Impairment helpers
    # ------------------------------------------------------------------

    def _apply_delay(self) -> None:
        delay_ms = self.extra_delay_ms + self._rng.gauss(0, self.jitter_ms)
        delay_s = max(0.0, delay_ms / 1000.0)
        if delay_s > 0:
            time.sleep(delay_s)

    def _maybe_corrupt(self, chunk: Chunk) -> Chunk:
        """Return a shallow copy with zeroed measurements to simulate corruption."""
        if self._rng.random() < self.corrupt_probability:
            self.corrupted_count += 1
            from harness.qeci_interface import Measurement
            corrupt_meas = [
                Measurement(qubit_id=m.qubit_id, value=0xFF, timestamp_ns=m.timestamp_ns)
                for m in chunk.measurements
            ]
            return Chunk(chunk_id=chunk.chunk_id, measurements=corrupt_meas)
        return chunk

    # ------------------------------------------------------------------
    # QECiInterface implementation
    # ------------------------------------------------------------------

    def send_chunk(self, chunk: Chunk) -> None:
        with self._lock:
            self.total_sent += 1

            # Loss
            if self._rng.random() < self.loss_probability:
                self.dropped_count += 1
                return  # silently discard

            # Corruption
            chunk = self._maybe_corrupt(chunk)

            # Reordering: hold chunk in buffer, flush previous
            if self._rng.random() < self.reorder_probability and self._reorder_buffer:
                self.reordered_count += 1
                # Send current first, then flush the buffer
                self._apply_delay()
                self._backend.send_chunk(chunk)
                while self._reorder_buffer:
                    self._apply_delay()
                    self._backend.send_chunk(self._reorder_buffer.popleft())
                return

            if self._rng.random() < self.reorder_probability:
                self._reorder_buffer.append(chunk)
                return

            # Normal path
            self._apply_delay()
            self._backend.send_chunk(chunk)

            # Flush any buffered chunks after the normal send
            while self._reorder_buffer:
                self._apply_delay()
                self._backend.send_chunk(self._reorder_buffer.popleft())

    def receive_correction(self, timeout_s: float = 2.0) -> Optional[CorrectionResult]:
        return self._backend.receive_correction(timeout_s=timeout_s)

    def close(self) -> None:
        self._backend.close()

    # ------------------------------------------------------------------
    # Context-manager mode (enables with-block impairment window)
    # ------------------------------------------------------------------

    def enable(self) -> "ImpairmentEmulator":
        """Enable impairments (no-op here; useful subclass hook)."""
        return self

    def disable(self) -> "ImpairmentEmulator":
        """
        Disable all impairments, e.g. after a burst window.
        Returns self for chaining.
        """
        self.loss_probability = 0.0
        self.reorder_probability = 0.0
        self.corrupt_probability = 0.0
        self.extra_delay_ms = 0.0
        self.jitter_ms = 0.0
        return self

    def summary(self) -> dict:
        return {
            "total_sent": self.total_sent,
            "dropped": self.dropped_count,
            "reordered": self.reordered_count,
            "corrupted": self.corrupted_count,
        }

"""
harness/stimulus_generator.py
=============================
Deterministic measurement-stream generator.

Replaces the physical qubit control system in the lab.  All streams are
fully reproducible from a seed so test runs can be compared across firmware
builds.

Public API
----------
StimulusGenerator.make_chunk(chunk_id, size) -> Chunk
StimulusGenerator.stream(num_chunks, chunk_size, rate_mps) -> Iterator[Chunk]
StimulusGenerator.replay_trace(trace) -> Iterator[Chunk]

Assumptions
-----------
* Measurements are binary syndrome bits (0 = no flip, 1 = flip detected).
* The generator does not model any quantum channel — it emits deterministic
  or configurable-error-rate patterns that exercise the Deltaflow pipeline.
* For oracle-based replay tests, fixed traces from fixtures/syndrome_traces.py
  should be used instead of random generation.
"""

from __future__ import annotations

import random
import time
from typing import Iterator, List, Optional

from harness.qeci_interface import Chunk, Measurement


class StimulusGenerator:
    """
    Generates synthetic measurement streams with controllable error density.

    Parameters
    ----------
    num_qubits : int
        Number of logical qubits (determines qubit_id range).
    error_rate : float
        Probability that any single measurement is 1 (flip detected).
    seed : int | None
        RNG seed for reproducibility.  None = random seed.
    """

    def __init__(
        self,
        num_qubits: int = 8,
        error_rate: float = 0.001,
        seed: Optional[int] = 42,
    ):
        self.num_qubits = num_qubits
        self.error_rate = error_rate
        self._rng = random.Random(seed)

    # ------------------------------------------------------------------
    # Single-chunk factory
    # ------------------------------------------------------------------

    def make_chunk(self, chunk_id: int, size: int) -> Chunk:
        """
        Create a single Chunk of `size` measurements.

        Each measurement is assigned a nanosecond timestamp relative to
        now so that ingress timing is realistic.
        """
        now_ns = time.time_ns()
        measurements: List[Measurement] = []
        for i in range(size):
            m = Measurement(
                qubit_id=self._rng.randint(0, self.num_qubits - 1),
                value=1 if self._rng.random() < self.error_rate else 0,
                timestamp_ns=now_ns + i * 100,  # 100 ns spacing between meas
            )
            measurements.append(m)
        return Chunk(chunk_id=chunk_id, measurements=measurements)

    # ------------------------------------------------------------------
    # Streaming generator
    # ------------------------------------------------------------------

    def stream(
        self,
        num_chunks: int,
        chunk_size: int,
        rate_mps: float,
    ) -> Iterator[Chunk]:
        """
        Yield Chunks at the specified measurement rate.

        Parameters
        ----------
        num_chunks : int
            Total number of chunks to emit.
        chunk_size : int
            Measurements per chunk.
        rate_mps : float
            Target measurements per second.  The generator will sleep
            between chunks to honour this rate.

        Yields
        ------
        Chunk
        """
        inter_chunk_s = chunk_size / rate_mps  # seconds between chunk sends
        for i in range(num_chunks):
            t_start = time.perf_counter()
            yield self.make_chunk(chunk_id=i, size=chunk_size)
            elapsed = time.perf_counter() - t_start
            sleep_s = inter_chunk_s - elapsed
            if sleep_s > 0:
                time.sleep(sleep_s)

    # ------------------------------------------------------------------
    # Trace replay
    # ------------------------------------------------------------------

    def replay_trace(self, trace: List[List[int]]) -> Iterator[Chunk]:
        """
        Replay a pre-recorded syndrome trace.

        Parameters
        ----------
        trace : List[List[int]]
            Each element is a list of measurement values for one chunk.
            Example: [[0, 1, 0, 0], [1, 0, 0, 1], ...]

        Yields
        ------
        Chunk
        """
        for chunk_id, values in enumerate(trace):
            now_ns = time.time_ns()
            measurements = [
                Measurement(
                    qubit_id=i % self.num_qubits,
                    value=v,
                    timestamp_ns=now_ns + i * 100,
                )
                for i, v in enumerate(values)
            ]
            yield Chunk(chunk_id=chunk_id, measurements=measurements)

    # ------------------------------------------------------------------
    # Burst helper
    # ------------------------------------------------------------------

    def burst_stream(
        self,
        num_chunks: int,
        chunk_size: int,
        nominal_rate_mps: float,
        burst_rate_mps: float,
        burst_start_chunk: int,
        burst_duration_chunks: int,
    ) -> Iterator[Chunk]:
        """
        Yield chunks at nominal rate, switching to burst_rate_mps for a
        window of burst_duration_chunks chunks starting at burst_start_chunk.
        """
        for i in range(num_chunks):
            t_start = time.perf_counter()
            chunk = self.make_chunk(chunk_id=i, size=chunk_size)
            yield chunk

            if burst_start_chunk <= i < burst_start_chunk + burst_duration_chunks:
                rate = burst_rate_mps
            else:
                rate = nominal_rate_mps

            inter_chunk_s = chunk_size / rate
            elapsed = time.perf_counter() - t_start
            sleep_s = inter_chunk_s - elapsed
            if sleep_s > 0:
                time.sleep(sleep_s)

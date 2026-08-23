"""
harness/qeci_interface.py
=========================
QECi transport abstraction layer.

Two concrete implementations are provided:

  QECiStub   — in-process stub that simulates Deltaflow's QECi behavior.
               Used when USE_REAL_HW = false.  All timing is synthetic but
               the API contract is identical to the real adapter.

  QECiAdapter — thin wrapper over a real network socket connected to the
               Deltaflow FPGA endpoint via the SFP / packet link.
               Used when USE_REAL_HW = true.

Both expose the same interface:
  send_chunk(chunk_id, measurements) -> None
  receive_correction(timeout) -> CorrectionResult | None
  close() -> None

Assumptions
-----------
* QECi is packet-based.  Each chunk is sent as one logical request packet.
* Correction results carry the originating chunk_id so ingress and egress
  events can be correlated without relying on ordering.
* For the stub, a configurable simulated decode latency models FPGA timing.
"""

from __future__ import annotations

import queue
import random
import socket
import struct
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import List, Optional

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class Measurement:
    """Single syndrome measurement from the (simulated) qubit array."""
    qubit_id: int
    value: int          # 0 or 1 (bit-flip detected?)
    timestamp_ns: int   # wall-clock at ingress (nanoseconds)


@dataclass
class Chunk:
    """A batch of measurements that Deltaflow processes as one unit."""
    chunk_id: int
    measurements: List[Measurement]
    final_measurement_time_ns: int = field(init=False)

    def __post_init__(self):
        if self.measurements:
            self.final_measurement_time_ns = max(
                m.timestamp_ns for m in self.measurements
            )
        else:
            self.final_measurement_time_ns = time.time_ns()


@dataclass
class CorrectionResult:
    """Correction result returned by Deltaflow over the QECi link."""
    chunk_id: int
    correction_ops: List[int]   # Pauli corrections per qubit (stub: random)
    egress_time_ns: int         # Timestamp when result arrived at test harness
    raw_payload: bytes = b""    # Raw QECi packet bytes (real HW only)


# ---------------------------------------------------------------------------
# Base interface
# ---------------------------------------------------------------------------

class QECiInterface:
    """Abstract base — do not instantiate directly."""

    def send_chunk(self, chunk: Chunk) -> None:
        raise NotImplementedError

    def receive_correction(self, timeout_s: float = 2.0) -> Optional[CorrectionResult]:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


# ---------------------------------------------------------------------------
# Software stub (no hardware required)
# ---------------------------------------------------------------------------

class QECiStub(QECiInterface):
    """
    In-process stub that simulates Deltaflow's QECi interface.

    The stub models:
      * Configurable base decode latency with Gaussian jitter.
      * Chunk-size-proportional processing time.
      * Optional parallel processing of multiple chunks.
      * Optional fault injection: dropped chunks, malformed results.
    """

    def __init__(
        self,
        base_latency_ms: float = 2.0,
        latency_jitter_ms: float = 0.3,
        latency_per_measurement_us: float = 10.0,
        drop_probability: float = 0.0,
        corrupt_probability: float = 0.0,
        max_parallel_chunks: int = 4,
    ):
        self.base_latency_ms = base_latency_ms
        self.latency_jitter_ms = latency_jitter_ms
        self.latency_per_measurement_us = latency_per_measurement_us
        self.drop_probability = drop_probability
        self.corrupt_probability = corrupt_probability
        self.max_parallel_chunks = max_parallel_chunks

        self._result_queue: queue.Queue[CorrectionResult] = queue.Queue()
        self._executor = ThreadPoolExecutor(
            max_workers=max_parallel_chunks,
            thread_name_prefix="QECiStubWorker",
        )
        self._closed = False

    def send_chunk(self, chunk: Chunk) -> None:
        """Submit a chunk for (simulated) decoding.  Non-blocking."""
        if self._closed:
            raise RuntimeError("QECiStub is closed")
        self._executor.submit(self._decode_chunk, chunk)

    def _decode_chunk(self, chunk: Chunk) -> None:
        # Simulated decode latency
        base = self.base_latency_ms / 1000.0
        jitter = random.gauss(0, self.latency_jitter_ms / 1000.0)
        per_meas = (
            len(chunk.measurements) * self.latency_per_measurement_us / 1e6
        )
        latency = max(0.0, base + jitter + per_meas)
        time.sleep(latency)

        # Fault injection: silently drop this chunk
        if random.random() < self.drop_probability:
            return

        n_qubits = max((m.qubit_id for m in chunk.measurements), default=0) + 1
        corrections = [random.randint(0, 1) for _ in range(n_qubits)]

        # Fault injection: corrupt the result
        if random.random() < self.corrupt_probability:
            corrections = []  # malformed payload

        result = CorrectionResult(
            chunk_id=chunk.chunk_id,
            correction_ops=corrections,
            egress_time_ns=time.time_ns(),
        )
        self._result_queue.put(result)

    def receive_correction(self, timeout_s: float = 2.0) -> Optional[CorrectionResult]:
        """Block until a correction result is available or timeout expires."""
        try:
            return self._result_queue.get(timeout=timeout_s)
        except queue.Empty:
            return None

    def close(self) -> None:
        self._closed = True
        self._executor.shutdown(wait=True)


# ---------------------------------------------------------------------------
# Real hardware adapter (skeleton — fill in protocol details)
# ---------------------------------------------------------------------------

class QECiAdapter(QECiInterface):
    """
    Network adapter for a real Deltaflow FPGA endpoint over the QECi
    packet-based link (SFP / 10G Ethernet).

    Wire format assumption (placeholder — replace with actual QECi framing):
      [4B chunk_id][2B num_measurements][N*2B measurement data][2B checksum]

    LIMITATIONS
    -----------
    * This skeleton does not implement real QECi framing — the encode/decode
      helpers below are placeholders.
    * No TLS / authentication — assumes a trusted lab network.
    * Single TCP connection; production use may require connection pooling.
    """

    HEADER_FMT = "!IH"   # big-endian: uint32 chunk_id, uint16 num_meas
    HEADER_SIZE = struct.calcsize(HEADER_FMT)

    def __init__(self, host: str, port: int, timeout_s: float = 2.0):
        self.host = host
        self.port = port
        self.timeout_s = timeout_s
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.settimeout(timeout_s)
        self._sock.connect((host, port))
        self._lock = threading.Lock()

    def _encode_chunk(self, chunk: Chunk) -> bytes:
        """Encode a Chunk into a QECi packet.  PLACEHOLDER."""
        meas_bytes = b"".join(
            struct.pack("!HB", m.qubit_id, m.value) for m in chunk.measurements
        )
        header = struct.pack(self.HEADER_FMT, chunk.chunk_id, len(chunk.measurements))
        # Real implementation: add QECi framing, preamble, CRC
        return header + meas_bytes

    def _decode_response(self, data: bytes) -> CorrectionResult:
        """Decode a QECi response packet.  PLACEHOLDER."""
        chunk_id = struct.unpack_from("!I", data, 0)[0]
        n_ops = (len(data) - 4) // 1
        ops = list(data[4 : 4 + n_ops])
        return CorrectionResult(
            chunk_id=chunk_id,
            correction_ops=ops,
            egress_time_ns=time.time_ns(),
            raw_payload=data,
        )

    def send_chunk(self, chunk: Chunk) -> None:
        payload = self._encode_chunk(chunk)
        with self._lock:
            self._sock.sendall(payload)

    def receive_correction(self, timeout_s: float = 2.0) -> Optional[CorrectionResult]:
        self._sock.settimeout(timeout_s)
        try:
            # Read fixed-size header first, then payload
            header = self._recv_exactly(self.HEADER_SIZE)
            if not header:
                return None
            chunk_id, n_ops = struct.unpack(self.HEADER_FMT, header)
            payload = self._recv_exactly(n_ops)
            return self._decode_response(header + payload)
        except (socket.timeout, ConnectionResetError):
            return None

    def _recv_exactly(self, n: int) -> bytes:
        buf = b""
        while len(buf) < n:
            chunk = self._sock.recv(n - len(buf))
            if not chunk:
                return buf
            buf += chunk
        return buf

    def close(self) -> None:
        try:
            self._sock.close()
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def create_interface(cfg: dict) -> QECiInterface:
    """
    Return the appropriate QECi interface based on the test config.

    cfg should be the full parsed YAML config dict (see config/test_config.yaml).
    """
    if cfg.get("deltaflow", {}).get("use_real_hw", False):
        host = cfg["deltaflow"]["host"]
        port = cfg["deltaflow"]["port"]
        timeout = cfg["deltaflow"]["response_timeout_s"]
        return QECiAdapter(host=host, port=port, timeout_s=timeout)
    else:
        return QECiStub(
            base_latency_ms=2.0,
            latency_jitter_ms=0.3,
            drop_probability=0.0,
        )

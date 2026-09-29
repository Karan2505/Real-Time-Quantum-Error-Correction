"""
Ingress pipeline building blocks of the AURA-Link gateway (FPGA side):

  AURA-Link PHY Rx  (phy.py: SerDes, block lock, descrambler, 64b/66b, CDC)
        |  SOF --------------------> ArrivalTimestampTagger
        |  header words (cut-through)
        v
  FrameParser  -- header check at word 4 (+pipeline), payload CRC at EOF
        v
  SequenceTracker -- gaps, duplicates, late frames, erasure timers
        v
  IngressElasticFifo -- bounded buffer, watermarks, credits, overflow policy
        v
  IngressSink (Layer 2: Router & Preprocessing Unit)

Every block is cycle-timed against the core clock of the FPGA (default
300 MHz) and keeps its own telemetry.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Callable, Optional, Protocol

from ..core.telemetry import MetricsRegistry
from .frames import (Header, HeaderError, check_payload, decode_header, frame_length, seq_diff)
from .phy import RxFrame
from .records import ErasureRecord, SyndromeRecord


# =========================================================================== timestamp tagger
class ArrivalTimestampTagger:
    """Hardware timestamp unit (TSU).

    Latches the (time-synchronised) local clock when the PCS reports a start
    of frame and refers it back to the wire by subtracting the constant RX
    PHY latency -- the same timestamp point IEEE 1588 uses.  The result is
    independent of frame length and of downstream queueing.
    """

    def __init__(self, clock, metrics: MetricsRegistry, name: str = "tsu", correction_ps: int = 0):
        self.clock = clock
        self.m = metrics
        self.name = name
        self.correction_ps = correction_ps     # extra calibrated asymmetry correction
        self.count = 0

    def on_sof(self, rxf: RxFrame) -> None:
        rxf.sof_ts = self.clock.timestamp() - rxf.rx_latency_ps - self.correction_ps
        self.count += 1
        self.m.counter(f"{self.name}.stamped").inc()


# =========================================================================== frame parser
@dataclass
class ParsedFrame:
    header: Header
    payload: bytes
    rxf: RxFrame
    t_header_ok: int
    t_eof_ok: int


class FrameParser:
    """Validates frames in two stages, timed on the core clock:

    * header stage (cut-through): as soon as bytes 0..35 are in the core
      domain -> preamble, header CRC-32C, version, length sanity.
      Costs ``hdr_cycles``.
    * end-of-frame stage: PCS status, exact length, payload CRC-32C.
      Costs ``eof_cycles`` after the last word.
    """

    def __init__(self, sched, core_period_ps: int, metrics: MetricsRegistry,
                 on_header: Callable[[Header, RxFrame], None],
                 on_frame: Callable[[ParsedFrame], None],
                 on_error: Callable[[str, Optional[Header], RxFrame], None],
                 hdr_cycles: int = 2, eof_cycles: int = 2, max_frame: int = 65535, name: str = "parser"):
        self.s = sched
        self.cp = core_period_ps
        self.m = metrics
        self.on_header_cb = on_header
        self.on_frame_cb = on_frame
        self.on_error_cb = on_error
        self.hdr_cycles = hdr_cycles
        self.eof_cycles = eof_cycles
        self.max_frame = max_frame
        self.name = name

    # -- stage 1 ---------------------------------------------------------------
    def header_available(self, rxf: RxFrame) -> None:
        self.s.call_later(self.hdr_cycles * self.cp, self._check_header, rxf)

    def _check_header(self, rxf: RxFrame) -> None:
        try:
            h = decode_header(rxf.header_bytes)
            if frame_length(h) > self.max_frame:
                raise HeaderError("oversize", "declared length exceeds maximum frame size")
        except HeaderError as e:
            rxf.ctx["header_error"] = e.reason
            return                                   # reported once at EOF (frame is discarded)
        rxf.ctx["header"] = h
        rxf.ctx["t_header_ok"] = self.s.now_ps()
        self.m.counter(f"{self.name}.headers_ok").inc()
        self.on_header_cb(h, rxf)

    # -- stage 2 ---------------------------------------------------------------
    def frame_complete(self, rxf: RxFrame) -> None:
        self.s.call_later(self.eof_cycles * self.cp, self._check_frame, rxf)

    def _check_frame(self, rxf: RxFrame) -> None:
        h: Optional[Header] = rxf.ctx.get("header")
        if "header_error" in rxf.ctx:
            self._error(rxf.ctx["header_error"], None, rxf)
            return
        if rxf.error:
            self._error("pcs:" + rxf.error, h, rxf)
            return
        data = rxf.data
        if h is None:
            # frame too short for early header extraction: parse the header now
            try:
                h = decode_header(data)
            except HeaderError as e:
                self._error(e.reason, None, rxf)
                return
            rxf.ctx["header"] = h
            rxf.ctx["t_header_ok"] = self.s.now_ps()
            self.m.counter(f"{self.name}.headers_ok").inc()
            self.on_header_cb(h, rxf)
        try:
            payload = check_payload(data, h)
        except HeaderError as e:
            self._error(e.reason, h, rxf)
            return
        self.m.counter(f"{self.name}.frames_ok").inc()
        self.on_frame_cb(ParsedFrame(h, payload, rxf, rxf.ctx.get("t_header_ok", self.s.now_ps()),
                                     self.s.now_ps()))

    def _error(self, reason: str, h: Optional[Header], rxf: RxFrame) -> None:
        self.m.counter(f"{self.name}.err.{reason}").inc()
        self.m.counter(f"{self.name}.errors").inc()
        self.on_error_cb(reason, h, rxf)


# =========================================================================== sequence tracker
class SequenceTracker:
    """Per-stream sequence accounting with erasure detection.

    * in-order frame                -> accepted
    * jump forward                  -> the skipped sequence numbers are *pending*;
                                       each is declared erased after ``reorder_timeout_ps``
                                       unless it arrives first (re-ordering tolerance)
    * pending number arrives        -> accepted (re-ordered)
    * already-erased number arrives -> accepted with ``late=True`` (policy) or dropped
    * already seen                  -> duplicate, dropped
    Sequence numbers use RFC 1982 serial arithmetic (wrap-safe).
    """

    def __init__(self, sched, metrics: MetricsRegistry, on_erasure: Callable[[ErasureRecord], None],
                 frames_per_shot: int, reorder_timeout_ps: int, window: int = 4096,
                 deliver_late: bool = True, name: str = "seq", patch: int = 0):
        self.s = sched
        self.m = metrics
        self.on_erasure = on_erasure
        self.fps = max(1, frames_per_shot)
        self.timeout = reorder_timeout_ps
        self.window = window
        self.deliver_late = deliver_late
        self.name = name
        self.patch = patch
        self.expected = 0
        self._seen: deque[int] = deque()
        self._seen_set: set[int] = set()
        self._pending: dict[int, object] = {}
        self._erased: set[int] = set()
        self._erased_order: deque[int] = deque()
        self._anchor: Optional[tuple[int, int, int]] = None     # (seq, shot, round) of a received frame

    def reset(self, start_seq: int = 0) -> None:
        for h in self._pending.values():
            self.s.cancel(h)
        self._pending.clear()
        self._erased.clear()
        self._erased_order.clear()
        self._seen.clear()
        self._seen_set.clear()
        self.expected = start_seq
        self._anchor = None

    def _remember(self, seq: int) -> None:
        self._seen.append(seq)
        self._seen_set.add(seq)
        while len(self._seen) > self.window:
            self._seen_set.discard(self._seen.popleft())

    def _infer(self, seq: int) -> tuple[int, int]:
        if self._anchor is None:
            return -1, -1
        a_seq, a_shot, a_round = self._anchor
        pos = a_round + seq_diff(seq, a_seq)
        return a_shot + pos // self.fps, pos % self.fps

    def observe(self, seq: int, shot: int, rnd: int) -> str:
        """Classify an arriving sequence number: 'ok', 'reordered', 'late', 'dup'."""
        if seq in self._seen_set:
            self.m.counter(f"{self.name}.duplicates").inc()
            return "dup"
        d = seq_diff(seq, self.expected)
        if d > self.window:
            # far outside the tracking window (e.g. the peer restarted its numbering):
            # resynchronise instead of arming an unbounded number of erasure timers
            self.m.counter(f"{self.name}.resync").inc()
            for h in self._pending.values():
                self.s.cancel(h)
            self._pending.clear()
            d = 0
            self.expected = seq
        if d >= 0:
            for miss in range(self.expected, self.expected + d):
                ms = miss & 0xFFFFFFFF
                self._pending[ms] = self.s.call_later(self.timeout, self._expire, ms)
                self.m.counter(f"{self.name}.gaps").inc()
            self.expected = (seq + 1) & 0xFFFFFFFF
            status = "ok"
        else:
            h = self._pending.pop(seq, None)
            if h is not None:
                self.s.cancel(h)
                self.m.counter(f"{self.name}.reordered").inc()
                status = "reordered"
            elif seq in self._erased:
                self._erased.discard(seq)
                self.m.counter(f"{self.name}.late").inc()
                status = "late" if self.deliver_late else "dup"
            else:
                # older than the window: treat as duplicate / stale
                self.m.counter(f"{self.name}.stale").inc()
                status = "dup"
        self._remember(seq)
        if status != "dup":
            self._anchor = (seq, shot, rnd)
        return status

    def mark_corrupt(self, seq: int) -> None:
        """Header was valid but the payload failed: the number is consumed (no gap erasure)."""
        self._remember(seq)

    def _expire(self, seq: int) -> None:
        if self._pending.pop(seq, None) is None:
            return
        self._erased.add(seq)
        self._erased_order.append(seq)
        while len(self._erased_order) > self.window:
            self._erased.discard(self._erased_order.popleft())
        shot, rnd = self._infer(seq)
        self.m.counter(f"{self.name}.erasures").inc()
        self.on_erasure(ErasureRecord(patch=self.patch, seq=seq, reason="gap", shot=shot, round=rnd,
                                      inferred=True, detect_ts=self.s.now_ps()))

    @property
    def pending(self) -> int:
        return len(self._pending)


# =========================================================================== elastic FIFO
class IngressSink(Protocol):
    """Interface implemented by Layer 2 (Router & Preprocessing Unit) or a stub."""

    def on_syndrome(self, rec: SyndromeRecord) -> None: ...

    def on_erasure(self, rec: ErasureRecord) -> None: ...

    def on_session(self, event: str, info: dict) -> None: ...


class IngressElasticFifo:
    """Bounded ingress buffer between the parser and the RPU.

    * ``capacity`` records; push when full -> overflow (record converted to an
      erasure so the decoder knows the round is missing);
    * ``service_ps``: minimum spacing between records handed to the sink
      (models the RPU's acceptance rate; 0 = sink always ready);
    * high / low watermarks drive the backpressure alarm;
    * cumulative ``popped`` count drives credit-based flow control.
    """

    def __init__(self, sched, metrics: MetricsRegistry, sink_fn: Callable[[object], None],
                 on_overflow: Callable[[SyndromeRecord], None], capacity: int = 256,
                 service_ps: int = 0, high_wm: float = 0.75, low_wm: float = 0.25,
                 core_period_ps: int = 3333, name: str = "fifo",
                 on_watermark: Optional[Callable[[bool], None]] = None,
                 on_pop: Optional[Callable[[int], None]] = None):
        if capacity < 1:
            raise ValueError("FIFO capacity must be >= 1")
        if not 0 < low_wm < high_wm <= 1:
            raise ValueError("need 0 < low_wm < high_wm <= 1")
        self.s = sched
        self.m = metrics
        self.sink_fn = sink_fn
        self.on_overflow = on_overflow
        self.capacity = capacity
        self.service_ps = service_ps
        self.high = int(high_wm * capacity)
        self.low = int(low_wm * capacity)
        self.cp = core_period_ps
        self.name = name
        self.on_watermark = on_watermark
        self.on_pop = on_pop
        self._q: deque = deque()
        self._next_free = 0
        self._drain = None
        self.pushed = 0
        self.popped = 0
        self.overflows = 0
        self.above_high = False
        self.max_occupancy = 0

    def __len__(self) -> int:
        return len(self._q)

    def push(self, rec) -> bool:
        if len(self._q) >= self.capacity:
            self.overflows += 1
            self.m.counter(f"{self.name}.overflows").inc()
            self.on_overflow(rec)
            return False
        self._q.append(rec)
        self.pushed += 1
        occ = len(self._q)
        self.max_occupancy = max(self.max_occupancy, occ)
        self.m.gauge(f"{self.name}.occupancy").set(occ)
        if not self.above_high and occ >= self.high:
            self.above_high = True
            self.m.counter(f"{self.name}.high_watermark").inc()
            if self.on_watermark:
                self.on_watermark(True)
        if self._drain is None:
            now = self.s.now_ps()
            # one core cycle to write + read through the FIFO
            delay = max(self.cp, self._next_free - now)
            self._drain = self.s.call_later(delay, self._pop)
        return True

    def _pop(self) -> None:
        self._drain = None
        if not self._q:
            return
        rec = self._q.popleft()
        self.popped += 1
        now = self.s.now_ps()
        if isinstance(rec, SyndromeRecord):
            rec.ready_ts = now
        occ = len(self._q)
        self.m.gauge(f"{self.name}.occupancy").set(occ)
        if self.above_high and occ <= self.low:
            self.above_high = False
            if self.on_watermark:
                self.on_watermark(False)
        self._next_free = now + max(self.service_ps, self.cp)
        self.sink_fn(rec)
        if self.on_pop:
            self.on_pop(self.popped)
        if self._q:
            self._drain = self.s.call_later(self._next_free - now, self._pop)

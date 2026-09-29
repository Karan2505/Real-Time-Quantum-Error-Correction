"""
AURA-PHY: the serial physical layer (SFP+/SFP28-class multi-gigabit link).

Transmit path   frame bytes -> 64b/66b PCS encode -> x^58+x^39+1 scrambler
                -> gearbox/serialiser (TX latency) -> fibre/twinax (propagation)
Receive path    CDR/deserialiser -> block lock -> descrambler -> 64b/66b decode
                (RX latency) -> async CDC FIFO (recovered clock -> FPGA core clock)

Modelled physics / hardware behaviour
-------------------------------------
* 66-bit blocks with 2-bit sync headers (01 data, 10 control) and IEEE 802.3
  clause-49 block-type codes for START / TERMINATE_k / IDLE.
* Self-synchronising scrambler on the 64-bit block payload (so a single line
  bit error becomes up to three data errors, exactly as on real links).
* Serial-line bit errors: uniform BER and bursty errors (Gilbert-Elliott), and
  loss-of-signal intervals (fibre pulls, retimer resets).
* Block-lock state machine: lock is lost after ``lock_bad_headers`` invalid
  sync headers within a 64-block window and re-acquired after
  ``relock_blocks`` consecutive good headers; frames received while unlocked
  are lost.
* Deterministic latencies: serialisation (66/64 overhead at the line rate),
  TX/RX SerDes pipelines, cable propagation, and a two-flop-synchroniser CDC
  into the core clock (arrival snapped to the next core-clock edge).
* Transmit QoS: strict-priority scheduler (real-time > timing > control >
  management) with a minimum inter-packet gap of one idle block.  Frames
  already on the wire are never pre-empted.

Idle blocks between frames are not simulated one by one (they carry no
information); their only effects -- descrambler resynchronisation and
lock maintenance -- are applied analytically.
"""

from __future__ import annotations

import heapq
import itertools
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from ..core.errors import LinkError
from ..core.telemetry import MetricsRegistry
from ..core.units import NS, S, ceil_to_grid

# ------------------------------------------------------------------------ PCS constants
SYNC_DATA = 0b01
SYNC_CTRL = 0b10
BT_IDLE = 0x1E
BT_START = 0x78
BT_TERM = (0x87, 0x99, 0xAA, 0xB4, 0xCC, 0xD2, 0xE1, 0xFF)    # T0..T7 (k data bytes)
_TERM_K = {t: k for k, t in enumerate(BT_TERM)}
_M64 = (1 << 64) - 1
_M58 = (1 << 58) - 1


@dataclass
class PhyConfig:
    line_rate_gbps: float = 10.3125          # SFP+ 10GBASE-R serial rate (baud)
    tx_latency_ps: int = 32 * NS             # PCS + gearbox + serialiser
    rx_latency_ps: int = 48 * NS             # CDR + deserialiser + block sync + descrambler + decode
    cable_m: float = 2.0
    prop_ps_per_m: int = 5000                # ~5 ns/m (fibre / twinax)
    scramble: bool = True
    lock_bad_headers: int = 16
    relock_blocks: int = 64
    core_clock_mhz: float = 300.0            # FPGA core clock of the receiving side
    cdc_sync_stages: int = 2
    core_phase_ps: int = 0                   # phase of the core clock edges
    ipg_blocks: int = 1                      # minimum idle blocks between frames

    @property
    def block_ps(self) -> int:
        return int(round(66 / (self.line_rate_gbps * 1e9) * S))

    @property
    def core_period_ps(self) -> int:
        return int(round(S / (self.core_clock_mhz * 1e6)))

    @property
    def prop_ps(self) -> int:
        return int(round(self.cable_m * self.prop_ps_per_m))

    def validate(self) -> None:
        if self.line_rate_gbps <= 0 or self.core_clock_mhz <= 0:
            raise LinkError("line rate and core clock must be positive")
        if self.cable_m < 0:
            raise LinkError("cable length must be >= 0")
        # the core must keep up with the line: 8 bytes / core cycle vs 8 bytes / block
        if self.core_period_ps > self.block_ps:
            raise LinkError("core clock too slow for the line rate with a 64-bit datapath")


@dataclass
class GilbertElliott:
    """Two-state burst-error channel evaluated per 66-bit block."""
    p_good_to_bad: float = 0.0
    p_bad_to_good: float = 0.5
    ber_bad: float = 1e-2


@dataclass
class Impairments:
    ber: float = 0.0                              # uniform bit-error rate
    burst: Optional[GilbertElliott] = None
    los_intervals: list[tuple[int, int]] = field(default_factory=list)   # loss of signal [t0, t1) true time

    def validate(self) -> None:
        if not 0.0 <= self.ber < 0.5:
            raise LinkError("BER must be in [0, 0.5)")
        for a, b in self.los_intervals:
            if b <= a:
                raise LinkError("loss-of-signal interval must have t1 > t0")


# ------------------------------------------------------------------------ PCS helpers
def pcs_encode(frame: bytes) -> list[tuple[int, int]]:
    """Frame bytes -> list of (sync_header, 64-bit payload) blocks (unscrambled)."""
    n = len(frame)
    if n < 7:
        raise LinkError("frame too short for PCS encoding")
    blocks = []
    first = BT_START | int.from_bytes(frame[0:7], "little") << 8
    blocks.append((SYNC_CTRL, first))
    i = 7
    while n - i >= 8:
        blocks.append((SYNC_DATA, int.from_bytes(frame[i:i + 8], "little")))
        i += 8
    k = n - i
    blocks.append((SYNC_CTRL, BT_TERM[k] | int.from_bytes(frame[i:n], "little") << 8))
    return blocks


def scramble(payloads: list[int], state: int) -> tuple[list[int], int]:
    """Self-synchronising scrambler x^58 + x^39 + 1 (per 64-bit payload, LSB first).
    ``state`` holds the last 58 transmitted (scrambled) bits, bit k = s[t-58+k]."""
    out = []
    h = state
    for d in payloads:
        # first 39 bits depend only on history
        lo = (d & ((1 << 39) - 1)) ^ ((h >> 19) & ((1 << 39) - 1)) ^ (h & ((1 << 39) - 1))
        h = ((h | (lo << 58)) >> 39) & _M58
        hi = ((d >> 39) & ((1 << 25) - 1)) ^ ((h >> 19) & ((1 << 25) - 1)) ^ (h & ((1 << 25) - 1))
        h = ((h | (hi << 58)) >> 25) & _M58
        out.append(lo | (hi << 39))
    return out, h


def descramble(payloads: list[int], state: int) -> tuple[list[int], int]:
    out = []
    h = state
    for s in payloads:
        lo_s = s & ((1 << 39) - 1)
        lo = lo_s ^ ((h >> 19) & ((1 << 39) - 1)) ^ (h & ((1 << 39) - 1))
        h = ((h | (lo_s << 58)) >> 39) & _M58
        hi_s = (s >> 39) & ((1 << 25) - 1)
        hi = hi_s ^ ((h >> 19) & ((1 << 25) - 1)) ^ (h & ((1 << 25) - 1))
        h = ((h | (hi_s << 58)) >> 25) & _M58
        out.append(lo | (hi << 39))
    return out, h


def pcs_decode(blocks: list[tuple[int, int]]) -> tuple[Optional[bytes], str]:
    """Decode descrambled blocks of one frame.  Returns (bytes, '') or (None, reason)."""
    if not blocks:
        return None, "empty"
    hdr, p = blocks[0]
    if hdr not in (SYNC_DATA, SYNC_CTRL):
        return None, "sync_header"
    if hdr != SYNC_CTRL or (p & 0xFF) != BT_START:
        return None, "no_start"
    out = bytearray((p >> 8).to_bytes(7, "little"))
    for j in range(1, len(blocks)):
        hdr, p = blocks[j]
        if hdr == SYNC_DATA:
            if j == len(blocks) - 1:
                return None, "no_terminate"
            out += p.to_bytes(8, "little")
            continue
        if hdr != SYNC_CTRL:
            return None, "sync_header"
        bt = p & 0xFF
        if bt in _TERM_K:
            if j != len(blocks) - 1:
                return None, "early_terminate"
            k = _TERM_K[bt]
            out += ((p >> 8) & ((1 << (8 * k)) - 1)).to_bytes(k, "little") if k else b""
            return bytes(out), ""
        return None, "bad_block_type"
    return None, "no_terminate"


# ------------------------------------------------------------------------ in-flight frame
@dataclass
class WireFrame:
    blocks: list[tuple[int, int]]        # scrambled blocks as transmitted
    scr_state_before: int                # scrambler history before the frame (idles resync RX)
    t_wire_sof: int                      # true time the first bit leaves the TX pin
    meta: dict = field(default_factory=dict)


@dataclass
class RxFrame:
    """A frame as delivered by the PHY into the FPGA core clock domain."""
    data: Optional[bytes]
    error: str                           # '' if the PCS decoded it cleanly
    t_sof_pcs: int                       # true time the START block left the PCS decoder
    t_wire_sof: int                      # true time the first bit arrived at the RX pin
    block_core_ts: list[int]             # true time each block became visible in the core domain
    rx_latency_ps: int                   # fixed RX latency to refer timestamps back to the wire
    nblocks: int
    header_bytes: Optional[bytes] = None # first 36 bytes, available at t_header_core (cut-through)
    t_header_core: int = -1
    sof_ts: int = -1                     # arrival timestamp (receiver clock, referred to the wire)
    ctx: dict = field(default_factory=dict)


class PhyTx:
    """Transmit PCS + serialiser with a strict-priority QoS scheduler."""

    PRIORITIES = {0: 0, 2: 1, 3: 2, 1: 3}   # channel class -> priority (0 = highest): RT, TIMING, CONTROL, MGMT

    def __init__(self, sched, cfg: PhyConfig, channel: "SerialChannel", metrics: MetricsRegistry, name: str = "tx"):
        cfg.validate()
        self.s = sched
        self.cfg = cfg
        self.ch = channel
        self.m = metrics
        self.name = name
        self._q: list = []
        self._seq = itertools.count()
        self._busy_until = 0
        self._scr = 0
        self._kick = None
        self.enabled = True

    def queue_depth(self) -> int:
        return len(self._q)

    def send(self, frame: bytes, channel_class: int = 0,
             on_wire_sof: Optional[Callable[[int], None]] = None) -> None:
        """Queue a frame.  ``on_wire_sof(t)`` fires (at time t) when its first bit hits the wire."""
        if not self.enabled:
            self.m.counter(f"{self.name}.dropped_disabled").inc()
            return
        prio = self.PRIORITIES.get(int(channel_class), 3)
        heapq.heappush(self._q, (prio, next(self._seq), frame, on_wire_sof))
        self.m.counter(f"{self.name}.queued").inc()
        self.m.gauge(f"{self.name}.queue_depth").set(len(self._q))
        if self._kick is None:
            self._kick = self.s.call_later(max(0, self._busy_until - self.s.now_ps()), self._serve)

    def _serve(self) -> None:
        self._kick = None
        if not self._q:
            return
        now = self.s.now_ps()
        if now < self._busy_until:
            self._kick = self.s.call_later(self._busy_until - now, self._serve)
            return
        _, _, frame, cb = heapq.heappop(self._q)
        blocks = pcs_encode(frame)
        cfg = self.cfg
        if cfg.scramble:
            scr, new_state = scramble([p for _, p in blocks], self._scr)
            before = self._scr
            self._scr = new_state
            blocks = [(h, p) for (h, _), p in zip(blocks, scr)]
        else:
            before = 0
        t_sof = now + cfg.tx_latency_ps
        nb = len(blocks)
        self._busy_until = now + (nb + cfg.ipg_blocks) * cfg.block_ps
        self.m.counter(f"{self.name}.frames").inc()
        self.m.counter(f"{self.name}.bytes").inc(len(frame))
        self.m.gauge(f"{self.name}.queue_depth").set(len(self._q))
        if cb is not None:
            self.s.call_later(cfg.tx_latency_ps, cb, t_sof)
        self.ch.transmit(WireFrame(blocks, before, t_sof))
        if self._q:
            self._kick = self.s.call_later(self._busy_until - now, self._serve)


class SerialChannel:
    """One direction of the serial link (fibre/twinax) with impairments."""

    def __init__(self, sched, cfg: PhyConfig, rng: np.random.Generator, metrics: MetricsRegistry,
                 impairments: Optional[Impairments] = None, name: str = "ch"):
        self.s = sched
        self.cfg = cfg
        self.rng = rng
        self.m = metrics
        self.imp = impairments or Impairments()
        self.imp.validate()
        self.name = name
        self.rx: Optional["PhyRx"] = None
        self._ge_bad = False

    def set_impairments(self, imp: Impairments) -> None:
        """Change the channel condition *now*.  If a loss of signal is cleared early,
        the receiver re-acquires block lock relock_blocks after the signal returns."""
        imp.validate()
        self.imp = imp
        if self.rx is not None:
            now = self.s.now_ps()
            relock_at = now + self.cfg.relock_blocks * self.cfg.block_ps
            still_down = [b for a, b in imp.los_intervals if a <= now < b]
            if still_down:
                relock_at = max(still_down) + self.cfg.relock_blocks * self.cfg.block_ps
            if self.rx.locked_from > relock_at:
                self.rx.locked_from = relock_at

    def los_active(self, t0: int, t1: int) -> bool:
        return any(a < t1 and t0 < b for a, b in self.imp.los_intervals)

    def _los_blocking(self, t0: int, t1: int) -> Optional[tuple[int, int]]:
        """LOS interval that overlaps [t0, t1) or whose re-lock period is still running at t0."""
        relock = self.cfg.relock_blocks * self.cfg.block_ps
        for a, b in self.imp.los_intervals:
            if a < t1 and t0 < b + relock:
                return a, b
        return None

    def transmit(self, wf: WireFrame) -> None:
        if self.rx is None:
            raise LinkError(f"{self.name}: channel has no receiver")
        cfg = self.cfg
        nb = len(wf.blocks)
        t_arr0 = wf.t_wire_sof + cfg.prop_ps
        t_end = t_arr0 + nb * cfg.block_ps
        los = self._los_blocking(t_arr0, t_end)
        if los is not None:
            # loss of signal: the receiver loses block lock at the start of the outage and
            # needs relock_blocks good blocks after it ends
            self.m.counter(f"{self.name}.los_lost").inc()
            self.rx.notify_los(los[0], los[1])
            return
        blocks = self._corrupt(wf.blocks)
        now = self.s.now_ps()
        # the START block is decoded one block time + RX latency after the first bit arrives;
        # the frame completes when its last block has been decoded
        t_sof_pcs = t_arr0 + cfg.block_ps + cfg.rx_latency_ps
        self.s.call_later(t_sof_pcs - now, self.rx.begin, blocks, wf, t_arr0)

    def _corrupt(self, blocks: list[tuple[int, int]]) -> list[tuple[int, int]]:
        imp = self.imp
        if imp.ber <= 0 and imp.burst is None:
            return blocks
        out = []
        flips = 0
        for hdr, p in blocks:
            ber = imp.ber
            if imp.burst is not None:
                ge = imp.burst
                if self._ge_bad:
                    if self.rng.random() < ge.p_bad_to_good:
                        self._ge_bad = False
                elif self.rng.random() < ge.p_good_to_bad:
                    self._ge_bad = True
                if self._ge_bad:
                    ber = max(ber, ge.ber_bad)
            if ber > 0:
                nerr = self.rng.binomial(66, ber)
                if nerr:
                    pos = self.rng.choice(66, size=nerr, replace=False)
                    for b in pos:
                        if b < 2:
                            hdr ^= (1 << b)
                        else:
                            p ^= (1 << int(b - 2))
                    flips += nerr
            out.append((hdr, p))
        if flips:
            self.m.counter(f"{self.name}.bit_errors").inc(flips)
        return out


class PhyRx:
    """Receive PCS: block lock, descrambling, 64b/66b decode and CDC into the core clock."""

    def __init__(self, sched, cfg: PhyConfig, metrics: MetricsRegistry, deliver: Callable[[RxFrame], None],
                 on_sof: Optional[Callable[[RxFrame], None]] = None,
                 on_header: Optional[Callable[[RxFrame], None]] = None, name: str = "rx"):
        cfg.validate()
        self.s = sched
        self.cfg = cfg
        self.m = metrics
        self.deliver = deliver
        self.on_sof = on_sof
        self.on_header = on_header
        self.name = name
        self.locked_from = 0            # true time from which block lock is held
        self.lock_losses = 0
        self._los_seen: set = set()

    @property
    def locked(self) -> bool:
        return self.s.now_ps() >= self.locked_from

    def notify_los(self, t_los_start: int, t_los_end: int) -> None:
        """Loss of signal [start, end): lock is re-acquired only after relock_blocks good blocks."""
        relock = t_los_end + self.cfg.relock_blocks * self.cfg.block_ps
        key = (t_los_start, t_los_end)
        if key not in self._los_seen:
            self._los_seen.add(key)
            self.lock_losses += 1
            self.m.counter(f"{self.name}.lock_lost").inc()
        self.locked_from = max(self.locked_from, relock)

    def _core_edge(self, t: int) -> int:
        cp = self.cfg.core_period_ps
        return ceil_to_grid(t - self.cfg.core_phase_ps, cp) + self.cfg.core_phase_ps

    def begin(self, blocks: list[tuple[int, int]], wf: WireFrame, t_arr0: int) -> None:
        """Runs when the START block leaves the PCS decoder.  Everything that happens to
        this frame later (header availability, completion, loss of lock) is scheduled at
        the exact time the hardware would observe it."""
        cfg = self.cfg
        bp = cfg.block_ps
        nb = len(blocks)
        now = self.s.now_ps()
        if t_arr0 < self.locked_from:
            self.m.counter(f"{self.name}.dropped_unlocked").inc()
            return
        if cfg.scramble:
            plain, _ = descramble([p for _, p in blocks], wf.scr_state_before)
            dblocks = [(h, p) for (h, _), p in zip(blocks, plain)]
        else:
            dblocks = list(blocks)
        h0, p0 = dblocks[0]
        if h0 != SYNC_CTRL or (p0 & 0xFF) != BT_START:
            self.m.counter(f"{self.name}.lost_no_start").inc()      # frame never recognised
            return
        t_pcs = [t_arr0 + (i + 1) * bp + cfg.rx_latency_ps for i in range(nb)]
        # --- block lock: >= k invalid sync headers inside a 64-block window -> lock lost
        bad = [i for i, (h, _) in enumerate(blocks) if h not in (SYNC_DATA, SYNC_CTRL)]
        lost_at = None
        if bad:
            self.m.counter(f"{self.name}.bad_sync_headers").inc(len(bad))
            k = cfg.lock_bad_headers
            for j in range(len(bad) - k + 1):
                if bad[j + k - 1] - bad[j] < 64:
                    lost_at = bad[j + k - 1]
                    break
        # --- CDC into the core clock domain -------------------------------------
        sync = cfg.cdc_sync_stages * cfg.core_period_ps
        core_ts = [self._core_edge(t + sync) for t in t_pcs]
        for i in range(1, nb):                   # FIFO order, one 64-bit word per core cycle
            core_ts[i] = max(core_ts[i], core_ts[i - 1] + cfg.core_period_ps)
        rxf = RxFrame(data=None, error="pending", t_sof_pcs=now, t_wire_sof=t_arr0, block_core_ts=core_ts,
                      rx_latency_ps=cfg.rx_latency_ps + bp, nblocks=nb)
        if self.on_sof is not None:
            self.on_sof(rxf)                     # timestamp unit latches here
        # --- header (cut-through): bytes 0..35 live in blocks 0..4 --------------------
        hb = 4
        if nb > hb and (lost_at is None or lost_at > hb) \
                and all(h == SYNC_DATA for h, _ in dblocks[1:hb + 1]):
            hdr = (p0 >> 8).to_bytes(7, "little") + b"".join(p.to_bytes(8, "little") for _, p in dblocks[1:hb + 1])
            rxf.header_bytes = hdr[:36]
            rxf.t_header_core = core_ts[hb]
            if self.on_header is not None:
                self.s.call_later(core_ts[hb] - now, self.on_header, rxf)
        # --- end of frame ------------------------------------------------------------
        if lost_at is not None:
            self.locked_from = max(self.locked_from, t_pcs[lost_at] + cfg.relock_blocks * bp)
            self.lock_losses += 1
            self.m.counter(f"{self.name}.lock_lost").inc()
            rxf.error = "lock_lost"
            rxf.block_core_ts = core_ts[:lost_at + 1]
            self.m.counter(f"{self.name}.pcs_errors").inc()
            self.s.call_later(core_ts[lost_at] - now, self.deliver, rxf)
            return
        data, err = pcs_decode(dblocks)
        rxf.data = data
        rxf.error = err
        if err:
            self.m.counter(f"{self.name}.pcs_errors").inc()
        self.m.counter(f"{self.name}.frames").inc()
        self.s.call_later(core_ts[-1] - now, self.deliver, rxf)

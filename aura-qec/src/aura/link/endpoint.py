"""
AURA-Link endpoints.

``LinkEndpoint``      common machinery: framing, per-stream sequence numbers,
                      receive pipeline (TSU -> parser -> dispatch), heartbeats
                      and liveness, reliable management transfers (go-back-N ARQ).
``ControlEndpoint``   control-system side: session initiator, time master,
                      SYN transmitter with credit accounting, COR receiver.
``IngressGateway``    FPGA accelerator side: session responder, time slave,
                      the full ingress pipeline
                        PHY Rx -> Frame Parser -> Arrival Timestamp Tagger
                        -> Sequence Tracker -> Ingress Elastic FIFO -> IngressSink
                      and the egress path for corrections.

Session state machine
---------------------
DOWN -> CONNECTING (HELLO sent, retried) -> [CONFIGURING: decoder config
pushed over MGMT if the gateway lacks it] -> SYNCING (time sync running) ->
UP (gateway clock locked; syndrome traffic accepted) ; any side -> DOWN on
liveness timeout or CLOSE.
"""

from __future__ import annotations

import enum
import struct
from dataclasses import dataclass, field
from typing import Callable, Optional


from ..core.clock import LocalClock
from ..core.errors import LinkError, ProtocolError
from ..core.telemetry import MetricsRegistry
from ..core.units import US
from . import frames as F
from .crc import crc32c
from .frames import FrameType, Header
from .ingress import (ArrivalTimestampTagger, FrameParser, IngressElasticFifo, IngressSink, ParsedFrame,
                      SequenceTracker)
from .phy import PhyConfig, RxFrame
from .records import (FLAG_LEAK, FLAG_SOFT, CorrectionRecord, ErasureRecord, SyndromeRecord)
from .timesync import TimeSyncConfig, TimeSyncMaster, TimeSyncSlave


class LinkState(enum.IntEnum):
    DOWN = 0
    CONNECTING = 1
    CONFIGURING = 2
    SYNCING = 3
    UP = 4
    FAILED = 5


@dataclass
class ClockConfig:
    offset_ps: int = 0
    freq_ppm: float = 0.0
    resolution_ps: int = 1000          # 1 ns timestamp resolution
    jitter_ps: float = 0.0
    wander_ppm_per_sqrt_s: float = 0.0


@dataclass
class LinkConfig:
    phy: PhyConfig = field(default_factory=PhyConfig)
    timesync: TimeSyncConfig = field(default_factory=TimeSyncConfig)
    heartbeat_ps: int = 50 * US
    liveness_timeout_ps: int = 250 * US
    hello_retry_ps: int = 30 * US
    hello_max_retries: int = 100
    reorder_timeout_ps: int = 2 * US
    deliver_late: bool = True
    fifo_capacity: int = 512
    fifo_service_ps: int = 0
    fifo_high_wm: float = 0.75
    fifo_low_wm: float = 0.25
    credit_update_every: int = 8
    mgmt_chunk: int = 1024
    mgmt_window: int = 8
    mgmt_rto_ps: int = 40 * US
    mgmt_max_retries: int = 30
    parser_hdr_cycles: int = 2
    parser_eof_cycles: int = 2
    require_time_lock: bool = True
    soft_bits: int = 4
    control_clock: ClockConfig = field(default_factory=ClockConfig)
    gateway_clock: ClockConfig = field(default_factory=lambda: ClockConfig(offset_ps=37_123_456, freq_ppm=12.5))

    @classmethod
    def for_udp(cls) -> "LinkConfig":
        """Timing parameters suited to software (OS) timestamping over IP networks."""
        from ..core.units import MS
        return cls(timesync=TimeSyncConfig(interval_ps=5 * MS, step_threshold_ps=5 * MS,
                                           lock_threshold_ps=500 * US, unlock_threshold_ps=5 * MS, lock_samples=3),
                   heartbeat_ps=20 * MS, liveness_timeout_ps=500 * MS, hello_retry_ps=50 * MS,
                   reorder_timeout_ps=20 * MS, mgmt_rto_ps=50 * MS, mgmt_window=16,
                   gateway_clock=ClockConfig(), control_clock=ClockConfig())

    def validate(self) -> None:
        self.phy.validate()
        if self.soft_bits not in (0, 4, 8):
            raise LinkError("soft_bits must be 0, 4 or 8")
        if self.liveness_timeout_ps <= self.heartbeat_ps:
            raise LinkError("liveness timeout must exceed the heartbeat period")
        if self.mgmt_chunk < 16 or self.mgmt_chunk > F.MAX_PAYLOAD - 32:
            raise LinkError("invalid MGMT chunk size")


# =========================================================================== MGMT ARQ
class _MgmtSender:
    def __init__(self, ep: "LinkEndpoint", xfer: int, kind: int, blob: bytes,
                 on_done: Callable[[bool, str], None]):
        self.ep = ep
        self.xfer = xfer
        self.kind = kind
        self.blob = blob
        self.total = len(blob)
        self.digest = crc32c(blob)
        self.on_done = on_done
        self.base = 0
        self.next = 0
        self.retries = 0
        self.timer = None
        self.done = False

    def start(self) -> None:
        self._pump()

    def _pump(self) -> None:
        c = self.ep.cfg
        while self.next < self.total and self.next < self.base + c.mgmt_window * c.mgmt_chunk:
            data = self.blob[self.next:self.next + c.mgmt_chunk]
            self.ep._send(FrameType.MGMT_DATA,
                          F.encode_mgmt(self.xfer, self.kind, self.total, self.next, self.digest, data))
            self.ep.m.counter("mgmt.chunks_sent").inc()
            self.next += len(data)
        if self.total == 0 and self.next == 0:
            self.ep._send(FrameType.MGMT_DATA, F.encode_mgmt(self.xfer, self.kind, 0, 0, self.digest, b""))
            self.next = 0
        self._arm()

    def _arm(self) -> None:
        self.ep.s.cancel(self.timer)
        self.timer = self.ep.s.call_later(self.ep.cfg.mgmt_rto_ps * (1 + min(self.retries, 4)), self._timeout)

    def _timeout(self) -> None:
        if self.done:
            return
        self.retries += 1
        self.ep.m.counter("mgmt.timeouts").inc()
        if self.retries > self.ep.cfg.mgmt_max_retries:
            self._finish(False, "too many retransmissions")
            return
        self.next = self.base              # go-back-N
        self._pump()

    def on_ack(self, status: int, next_off: int) -> None:
        if self.done:
            return
        if status == 2:
            self._finish(False, "receiver rejected transfer (digest mismatch)")
            return
        if next_off > self.base:
            self.base = min(next_off, self.total)
            self.retries = 0
        if self.base >= self.total and (status == 1 or self.total == 0 or next_off >= self.total):
            self._finish(True, "")
            return
        self._pump()

    def _finish(self, ok: bool, why: str) -> None:
        self.done = True
        self.ep.s.cancel(self.timer)
        self.on_done(ok, why)


class _MgmtReceiver:
    def __init__(self, ep: "LinkEndpoint"):
        self.ep = ep
        self.bufs: dict[int, dict] = {}

    def on_data(self, xfer: int, kind: int, total: int, off: int, digest: int, data: bytes) -> None:
        st = self.bufs.get(xfer)
        if st is None:
            st = self.bufs[xfer] = {"buf": bytearray(), "total": total, "digest": digest, "done": False}
        if st["done"]:
            self.ep._send(FrameType.MGMT_ACK, F.encode_mgmt_ack(xfer, 1, st["total"]))
            return
        if off == len(st["buf"]) and total == st["total"]:
            st["buf"] += data
        else:
            self.ep.m.counter("mgmt.out_of_order").inc()
        status = 0
        if len(st["buf"]) >= st["total"]:
            blob = bytes(st["buf"][:st["total"]])
            if crc32c(blob) != st["digest"]:
                status = 2
                del self.bufs[xfer]
                self.ep.m.counter("mgmt.digest_fail").inc()
            else:
                status = 1
                st["done"] = True
                self.ep.m.counter("mgmt.transfers_ok").inc()
                self.ep._on_mgmt_complete(kind, blob)
        self.ep._send(FrameType.MGMT_ACK, F.encode_mgmt_ack(xfer, status, len(st["buf"]) if status != 2 else 0))


# =========================================================================== base endpoint
class LinkEndpoint:
    ROLE = "base"

    def __init__(self, sched, cfg: LinkConfig, clock: LocalClock, metrics: Optional[MetricsRegistry] = None,
                 name: str = "ep"):
        cfg.validate()
        self.s = sched
        self.cfg = cfg
        self.clock = clock
        self.name = name
        self.m = metrics or MetricsRegistry(name)
        self.tx = None                         # object with .send(bytes, channel_class, on_wire_sof)
        self.state = LinkState.DOWN
        self.epoch = 0
        self._seqs: dict[tuple[int, int], int] = {}
        self.tsu = ArrivalTimestampTagger(clock, self.m, name="tsu")
        self.parser = FrameParser(sched, cfg.phy.core_period_ps, self.m, self._on_header, self._on_frame,
                                  self._on_parse_error, cfg.parser_hdr_cycles, cfg.parser_eof_cycles)
        self._last_rx = -1
        self._hb_timer = None
        self._live_timer = None
        self._mgmt_rx = _MgmtReceiver(self)
        self._mgmt_tx: dict[int, _MgmtSender] = {}
        self._next_xfer = 1
        self.frames_tx = 0
        self.frames_rx = 0
        self.rx_errors = 0
        self.state_log: list[tuple[int, str]] = []
        self.listeners: list[Callable[[str, dict], None]] = []

    # ------------------------------------------------------------ wiring
    def attach_tx(self, tx) -> None:
        self.tx = tx

    # PHY Rx callbacks (the PHY calls these in hardware order)
    def phy_sof(self, rxf: RxFrame) -> None:
        self.tsu.on_sof(rxf)

    def phy_header(self, rxf: RxFrame) -> None:
        self.parser.header_available(rxf)

    def phy_deliver(self, rxf: RxFrame) -> None:
        self.parser.frame_complete(rxf)

    # ------------------------------------------------------------ state
    def _set_state(self, st: LinkState, why: str = "") -> None:
        if st == self.state:
            return
        self.state_log.append((self.s.now_ps(), f"{self.state.name}->{st.name} {why}".strip()))
        self.m.counter(f"state.{st.name}").inc()
        self.state = st
        for fn in self.listeners:
            fn("state", {"state": st.name, "why": why})
        self._on_state(st)

    def _on_state(self, st: LinkState) -> None:
        pass

    # ------------------------------------------------------------ transmit
    def _next_seq(self, ftype: int, patch: int) -> int:
        k = (int(ftype), patch)
        v = self._seqs.get(k, 0)
        self._seqs[k] = (v + 1) & 0xFFFFFFFF
        return v

    def _reset_seqs(self) -> None:
        self._seqs.clear()

    def _send(self, ftype: FrameType, payload: bytes = b"", patch: int = 0, shot: int = 0, rnd: int = 0,
              flags: int = 0, origin_ts: Optional[int] = None, on_sof=None, seq: Optional[int] = None) -> int:
        if self.tx is None:
            raise LinkError(f"{self.name}: no transmitter attached")
        s = self._next_seq(ftype, patch) if seq is None else seq
        h = Header(type=int(ftype), flags=flags, channel=int(F.CHANNEL_OF[ftype]), patch=patch, seq=s,
                   shot=shot & 0xFFFFFFFF, round=rnd, origin_ts=(self.clock.now_ps() if origin_ts is None
                                                               else origin_ts) & ((1 << 64) - 1),
                   epoch=self.epoch)
        frame = F.encode_frame(h, payload)
        self.tx.send(frame, int(F.CHANNEL_OF[ftype]), on_sof)
        self.frames_tx += 1
        self.m.counter(f"tx.{FrameType(ftype).name}").inc()
        return s

    # ------------------------------------------------------------ receive
    def _on_header(self, h: Header, rxf: RxFrame) -> None:
        self._touch()
        if h.type == FrameType.SYN:
            self._syn_header(h, rxf)

    def _syn_header(self, h: Header, rxf: RxFrame) -> None:
        pass

    def _on_parse_error(self, reason: str, h: Optional[Header], rxf: RxFrame) -> None:
        self.rx_errors += 1
        if h is not None:
            self._touch()
        self._rx_error(reason, h, rxf)

    def _rx_error(self, reason: str, h: Optional[Header], rxf: RxFrame) -> None:
        pass

    def _touch(self) -> None:
        self._last_rx = self.s.now_ps()

    def _on_frame(self, pf: ParsedFrame) -> None:
        self.frames_rx += 1
        h = pf.header
        try:
            ft = FrameType(h.type)
        except ValueError:
            self.m.counter("rx.unknown_type").inc()
            return
        self.m.counter(f"rx.{ft.name}").inc()
        try:
            self._dispatch(ft, h, pf)
        except (ProtocolError, F.HeaderError) as e:
            self.m.counter("rx.protocol_errors").inc()
            self.m.counter(f"rx.protocol_error.{getattr(e, 'reason', 'other')}").inc()

    def _dispatch(self, ft: FrameType, h: Header, pf: ParsedFrame) -> None:
        p = pf.payload
        if ft == FrameType.HEARTBEAT:
            if self._no_session_hb():
                return
            self._on_heartbeat(h, *F.decode_heartbeat(p))
        elif ft == FrameType.MGMT_DATA:
            self._mgmt_rx.on_data(*F.decode_mgmt(p))
        elif ft == FrameType.MGMT_ACK:
            x, st, nx = F.decode_mgmt_ack(p)
            snd = self._mgmt_tx.get(x)
            if snd is not None:
                snd.on_ack(st, nx)
        elif ft == FrameType.CLOSE:
            self._on_peer_close(p.decode(errors="replace"))
        else:
            self._dispatch_role(ft, h, pf)

    def _on_peer_close(self, reason: str) -> None:
        self._stop_keepalive()
        self._set_state(LinkState.DOWN, f"peer closed ({reason})" if reason else "peer closed")

    def _dispatch_role(self, ft: FrameType, h: Header, pf: ParsedFrame) -> None:
        self.m.counter("rx.unexpected").inc()

    def _no_session_hb(self) -> bool:
        return False

    # ------------------------------------------------------------ heartbeat / liveness
    def _start_keepalive(self) -> None:
        self.s.cancel(self._hb_timer)
        self.s.cancel(self._live_timer)
        self._last_rx = self.s.now_ps()
        self._hb()
        self._live_check()

    def _stop_keepalive(self) -> None:
        self.s.cancel(self._hb_timer)
        self.s.cancel(self._live_timer)
        self._hb_timer = self._live_timer = None

    def _hb(self) -> None:
        self._send(FrameType.HEARTBEAT, F.encode_heartbeat(int(self.state), self._credit_limit(),
                                                           self.frames_tx, self.frames_rx, self.rx_errors))
        self._hb_timer = self.s.call_later(self.cfg.heartbeat_ps, self._hb)

    def _live_check(self) -> None:
        now = self.s.now_ps()
        if now - self._last_rx > self.cfg.liveness_timeout_ps:
            self.m.counter("liveness_timeouts").inc()
            self._on_liveness_lost()
            return
        self._live_timer = self.s.call_later(self.cfg.heartbeat_ps, self._live_check)

    def _on_liveness_lost(self) -> None:
        self._stop_keepalive()
        self._set_state(LinkState.DOWN, "liveness timeout")

    def _credit_limit(self) -> int:
        return 0xFFFFFFFF

    def _on_heartbeat(self, h: Header, state: int, credit: int, tx: int, rx: int, err: int) -> None:
        pass

    # ------------------------------------------------------------ management transfers
    def send_blob(self, kind: int, blob: bytes, on_done: Callable[[bool, str], None]) -> int:
        x = self._next_xfer
        self._next_xfer = (self._next_xfer + 1) & 0xFFFF or 1
        snd = _MgmtSender(self, x, kind, blob, lambda ok, why, x=x: (self._mgmt_tx.pop(x, None), on_done(ok, why)))
        self._mgmt_tx[x] = snd
        snd.start()
        return x

    def _on_mgmt_complete(self, kind: int, blob: bytes) -> None:
        pass

    # ------------------------------------------------------------ misc
    def close(self) -> None:
        self._closing = True
        if self.tx is not None and self.state != LinkState.DOWN:
            self._send(FrameType.CLOSE, b"local close")
        self._stop_keepalive()
        for attr in ("ts_master",):
            if hasattr(self, attr):
                getattr(self, attr).stop()
        self.s.cancel(getattr(self, "_reconnect_timer", None))
        self._set_state(LinkState.DOWN, "local close")

    def stats(self) -> dict:
        snap = self.m.snapshot()
        snap["state"] = self.state.name
        snap["epoch"] = self.epoch
        snap["frames_tx"] = self.frames_tx
        snap["frames_rx"] = self.frames_rx
        snap["rx_errors"] = self.rx_errors
        return snap


# =========================================================================== control side
MGMT_KIND_DECODER_CONFIG = 1


class ControlEndpoint(LinkEndpoint):
    """Control-system side of AURA-Link (session initiator, time master)."""

    ROLE = "control"

    def __init__(self, sched, cfg: LinkConfig, clock: LocalClock, metrics: Optional[MetricsRegistry] = None,
                 name: str = "control-link"):
        super().__init__(sched, cfg, clock, metrics, name)
        self.ts_master = TimeSyncMaster(sched, clock, self._ts_send, cfg.timesync, self.m)
        self.on_correction: Optional[Callable[[CorrectionRecord], None]] = None
        self._manifest_blob = b""
        self._patches: list[tuple[int, int, int]] = []
        self.frames_per_shot = 1
        self._hello_timer = None
        self._hello_tries = 0
        self.peer_state = LinkState.DOWN
        self.credit_limit = 0
        self.syn_sent = 0
        self.credit_overruns = 0
        self._cor_trackers: dict[int, SequenceTracker] = {}
        self.on_ready: Optional[Callable[[], None]] = None
        self.auto_reconnect = True
        self._reconnect_timer = None
        self._closing = False

    def _ts_send(self, ftype, payload, on_sof) -> None:
        self._send(ftype, payload, on_sof=on_sof)

    # ------------------------------------------------------------ session
    def open(self, manifest) -> None:
        """Start a session for a compiled experiment (``DecoderManifest``)."""
        self._closing = False
        self._manifest_blob = manifest.to_blob()
        self.epoch = manifest.config_epoch()
        self._patches = [(pm.patch_id, pm.rounds + 1, max(len(a) for a in pm.round_acqs))
                         for pm in manifest.patches]
        self.frames_per_shot = sum(fp for _, fp, _ in self._patches)
        self._begin_session()

    def _begin_session(self) -> None:
        self.ts_master.stop()
        self._stop_keepalive()
        self.s.cancel(self._hello_timer)
        self._cor_trackers = {pid: SequenceTracker(self.s, self.m, lambda e: self.m.counter("cor.erasures").inc(),
                                                   1, self.cfg.reorder_timeout_ps * 10, name=f"cor{pid}", patch=pid)
                              for pid, _, _ in self._patches}
        self._reset_seqs()
        self.syn_sent = 0
        self.credit_limit = 0
        self._hello_tries = 0
        self._set_state(LinkState.CONNECTING, "open")
        self._send_hello()

    def _send_hello(self) -> None:
        if self.state not in (LinkState.CONNECTING,):
            return
        self._hello_tries += 1
        if self._hello_tries > self.cfg.hello_max_retries:
            self._set_state(LinkState.FAILED, "no HELLO_ACK")
            return
        patches = b"".join(struct.pack("<HHH", p, f, n) for p, f, n in self._patches)
        caps = (F.CAP_SOFT4 | F.CAP_SOFT8 | F.CAP_LEAK | F.CAP_CUT_THROUGH | F.CAP_MGMT_ARQ |
                F.CAP_TIMESYNC | F.CAP_ERASURE | F.CAP_CREDITS)
        tlv = F.encode_tlv({F.TLV_ROLE: b"\x00", F.TLV_NODE: self.name.encode(),
                            F.TLV_CAPS: struct.pack("<I", caps), F.TLV_PATCHES: patches,
                            F.TLV_MAX_FRAME: struct.pack("<H", 65535)})
        self._send(FrameType.HELLO, tlv)
        self._hello_timer = self.s.call_later(self.cfg.hello_retry_ps, self._send_hello)

    def _dispatch_role(self, ft: FrameType, h: Header, pf: ParsedFrame) -> None:
        p = pf.payload
        if ft == FrameType.HELLO_ACK:
            self._on_hello_ack(h, F.decode_tlv(p))
        elif ft == FrameType.TS_DELAY_REQ:
            seq, _ = F.decode_ts(p)
            self.ts_master.on_delay_req(seq, pf.rxf.sof_ts)
        elif ft == FrameType.CREDIT:
            limit, occ, _ = F.decode_credit(p)
            self._update_credit(limit)
            self.m.gauge("peer_fifo_occupancy").set(occ)
        elif ft == FrameType.COR:
            self._on_cor(h, pf)
        else:
            self.m.counter("rx.unexpected").inc()

    def _on_hello_ack(self, h: Header, tlv: dict) -> None:
        if self.state not in (LinkState.CONNECTING, LinkState.CONFIGURING):
            return
        status = tlv.get(F.TLV_STATUS, b"\xff")[0]
        self.s.cancel(self._hello_timer)
        if status == F.HELLO_OK:
            self._update_credit(struct.unpack("<I", tlv[F.TLV_CREDIT])[0] if F.TLV_CREDIT in tlv else 0, force=True)
            self._set_state(LinkState.SYNCING, "hello ok")
            self._start_keepalive()
            self.ts_master.start()
        elif status == F.HELLO_NEED_CONFIG:
            if self.state == LinkState.CONFIGURING:
                return
            self._set_state(LinkState.CONFIGURING, "pushing decoder config")
            self._start_keepalive()
            self.send_blob(MGMT_KIND_DECODER_CONFIG, self._manifest_blob, self._config_pushed)
        else:
            why = tlv.get(F.TLV_REASON, b"").decode(errors="replace")
            self._set_state(LinkState.FAILED, f"HELLO rejected ({status}): {why}")

    def _config_pushed(self, ok: bool, why: str) -> None:
        if not ok:
            self._set_state(LinkState.FAILED, f"config transfer failed: {why}")
            return
        self._stop_keepalive()
        self._set_state(LinkState.CONNECTING, "config delivered")
        self._hello_tries = 0
        self._send_hello()

    def _on_heartbeat(self, h: Header, state: int, credit: int, tx: int, rx: int, err: int) -> None:
        try:
            self.peer_state = LinkState(state)
        except ValueError:
            return
        if credit != 0xFFFFFFFF:
            self._update_credit(credit)
        if self.state == LinkState.SYNCING and self.peer_state == LinkState.UP:
            self._set_state(LinkState.UP, "gateway time-locked")
            if self.on_ready:
                self.on_ready()
        elif self.state == LinkState.UP and self.peer_state != LinkState.UP:
            self._set_state(LinkState.SYNCING, f"gateway reports {self.peer_state.name}")

    def _on_liveness_lost(self) -> None:
        self.ts_master.stop()
        super()._on_liveness_lost()
        self._schedule_reconnect()

    def _on_peer_close(self, reason: str) -> None:
        if self.state == LinkState.DOWN and reason == "no session" and self._reconnect_timer is not None:
            return
        self.ts_master.stop()
        super()._on_peer_close(reason)
        self._schedule_reconnect()

    def _schedule_reconnect(self) -> None:
        """Automatic session recovery: re-run the handshake with the same configuration.
        The control system pauses at the next shot boundary until the link is UP again."""
        if not self.auto_reconnect or not self._manifest_blob or self._closing:
            return
        self.s.cancel(self._reconnect_timer)
        self.m.counter("reconnects").inc()
        self._reconnect_timer = self.s.call_later(self.cfg.hello_retry_ps, self._reconnect)

    def _reconnect(self) -> None:
        self._reconnect_timer = None
        if self.state in (LinkState.DOWN, LinkState.FAILED):
            self._begin_session()

    def _update_credit(self, limit: int, force: bool = False) -> None:
        if force or F.seq_diff(limit, self.credit_limit) > 0:
            self.credit_limit = limit & 0xFFFFFFFF

    @property
    def credits_available(self) -> int:
        return F.seq_diff(self.credit_limit, self.syn_sent & 0xFFFFFFFF)

    # ------------------------------------------------------------ SyndromeSink API
    def can_start_shot(self) -> bool:
        return self.state == LinkState.UP and self.credits_available >= self.frames_per_shot

    def send_syndrome(self, rec: SyndromeRecord) -> None:
        if self.state != LinkState.UP:
            self.m.counter("syn.dropped_link_not_up").inc()
            return
        if self.credits_available <= 0:
            # real-time data cannot wait: transmit and account the overrun
            self.credit_overruns += 1
            self.m.counter("syn.credit_overrun").inc()
        sb = self.cfg.soft_bits if rec.soft is not None else 0
        flags = rec.flags
        soft = rec.soft if sb else None
        if not sb:
            flags &= ~FLAG_SOFT
        leak = rec.leak if (flags & FLAG_LEAK) and rec.leak is not None else None
        if leak is None:
            flags &= ~FLAG_LEAK
        payload = F.encode_syn_payload(rec.bits, soft, sb, leak)
        rec.seq = self._send(FrameType.SYN, payload, patch=rec.patch, shot=rec.shot, rnd=rec.round,
                             flags=flags, origin_ts=rec.origin_ts)
        self.syn_sent += 1

    # ------------------------------------------------------------ corrections
    def _on_cor(self, h: Header, pf: ParsedFrame) -> None:
        if h.epoch != self.epoch:
            self.m.counter("cor.epoch_mismatch").inc()
            return
        tr = self._cor_trackers.get(h.patch)
        if tr is not None and tr.observe(h.seq, h.shot, h.round) == "dup":
            return
        obs, status, lat, ref = F.decode_cor_payload(pf.payload)
        cor = CorrectionRecord(patch=h.patch, shot=h.shot, round=h.round, obs_flips=obs, status=status,
                               decoder_latency_ns=lat, ref_seq=ref, origin_ts=h.origin_ts, seq=h.seq,
                               arrival_ts=pf.rxf.sof_ts)
        if self.state == LinkState.UP:
            self.m.histogram("cor.transport_ps").record(max(0, cor.arrival_ts - cor.origin_ts))
        if self.on_correction:
            self.on_correction(cor)


# =========================================================================== gateway
class IngressGateway(LinkEndpoint):
    """AURA-Link ingress transport gateway (FPGA accelerator side)."""

    ROLE = "gateway"

    def __init__(self, sched, cfg: LinkConfig, clock: LocalClock, sink: Optional[IngressSink] = None,
                 metrics: Optional[MetricsRegistry] = None, name: str = "gateway"):
        super().__init__(sched, cfg, clock, metrics, name)
        self.sink = sink
        self.ts_slave = TimeSyncSlave(sched, clock, self._ts_send, cfg.timesync, self.m, self._on_lock_change)
        self.fifo = IngressElasticFifo(sched, self.m, self._to_sink, self._fifo_overflow,
                                       capacity=cfg.fifo_capacity, service_ps=cfg.fifo_service_ps,
                                       high_wm=cfg.fifo_high_wm, low_wm=cfg.fifo_low_wm,
                                       core_period_ps=cfg.phy.core_period_ps, on_pop=self._on_pop,
                                       on_watermark=self._on_watermark)
        self.manifests: dict[int, object] = {}
        self.manifest = None
        self._trackers: dict[int, SequenceTracker] = {}
        self._round_sizes: dict[int, list[int]] = {}
        self._credit_base = 0
        self._last_credit_sent = 0
        self._pending_epoch = 0
        self._last_close_ps = -(1 << 62)

    def _ts_send(self, ftype, payload, on_sof) -> None:
        self._send(ftype, payload, on_sof=on_sof)

    # ------------------------------------------------------------ provisioning
    def provision(self, manifest) -> None:
        """Pre-load a decoder configuration (otherwise it is pushed over MGMT)."""
        self.manifests[manifest.config_epoch()] = manifest

    def _on_mgmt_complete(self, kind: int, blob: bytes) -> None:
        if kind != MGMT_KIND_DECODER_CONFIG:
            self.m.counter("mgmt.unknown_kind").inc()
            return
        from ..host.compiler.manifest import DecoderManifest
        try:
            man = DecoderManifest.from_blob(blob)
        except Exception:
            self.m.counter("mgmt.bad_config").inc()
            return
        ep = man.config_epoch()
        self.manifests[ep] = man
        self.m.counter("config_received").inc()

    # ------------------------------------------------------------ session (responder)
    def _no_session_hb(self) -> bool:
        return self._no_session(FrameType.HEARTBEAT)

    def _no_session(self, ft: FrameType) -> bool:
        """While DOWN, answer session traffic with a (rate-limited) CLOSE so the peer
        cannot remain in a half-open session; it will re-handshake."""
        if self.state != LinkState.DOWN or ft in (FrameType.HELLO, FrameType.MGMT_DATA, FrameType.MGMT_ACK,
                                                  FrameType.CLOSE):
            return False
        now = self.s.now_ps()
        if now - self._last_close_ps >= self.cfg.heartbeat_ps:
            self._last_close_ps = now
            self._send(FrameType.CLOSE, b"no session")
            self.m.counter("close_no_session").inc()
        return True

    def _dispatch_role(self, ft: FrameType, h: Header, pf: ParsedFrame) -> None:
        p = pf.payload
        if self._no_session(ft):
            return
        if ft == FrameType.HELLO:
            self._on_hello(h, F.decode_tlv(p))
        elif ft == FrameType.TS_SYNC:
            seq, _ = F.decode_ts(p)
            self.ts_slave.on_sync(seq, pf.rxf.sof_ts)
        elif ft == FrameType.TS_FOLLOW_UP:
            self.ts_slave.on_follow_up(*F.decode_ts(p))
        elif ft == FrameType.TS_DELAY_RESP:
            self.ts_slave.on_delay_resp(*F.decode_ts(p))
        elif ft == FrameType.SYN:
            self._on_syn(h, pf)
        else:
            self.m.counter("rx.unexpected").inc()

    def _hello_ack(self, status: int, reason: str = "") -> None:
        items = {F.TLV_ROLE: b"\x01", F.TLV_NODE: self.name.encode(), F.TLV_STATUS: bytes([status])}
        if status == F.HELLO_OK:
            items[F.TLV_CREDIT] = struct.pack("<I", self._credit_limit())
        if reason:
            items[F.TLV_REASON] = reason.encode()
        self._send(FrameType.HELLO_ACK, F.encode_tlv(items))

    def _on_hello(self, h: Header, tlv: dict) -> None:
        if h.version >> 4 != F.VERSION_MAJOR:
            self._hello_ack(F.HELLO_VERSION, "version mismatch")
            return
        if h.epoch == self.epoch and self.manifest is not None and \
                self.state in (LinkState.SYNCING, LinkState.UP):
            # duplicate HELLO (our ACK was lost or crossed a retry): re-acknowledge, keep state
            self.m.counter("hello_duplicate").inc()
            self._hello_ack(F.HELLO_OK)
            return
        man = self.manifests.get(h.epoch)
        if man is None:
            self._hello_ack(F.HELLO_NEED_CONFIG)
            if self.state == LinkState.DOWN:
                self._start_keepalive()
            return
        # validate the announced stream layout against the configuration
        pat = tlv.get(F.TLV_PATCHES, b"")
        announced = {struct.unpack_from("<HHH", pat, i)[0]: struct.unpack_from("<HHH", pat, i)[1:]
                     for i in range(0, len(pat) - len(pat) % 6, 6)}
        for pm in man.patches:
            fp, nm = announced.get(pm.patch_id, (0, 0))
            if fp != pm.rounds + 1:
                self._hello_ack(F.HELLO_EPOCH_MISMATCH, f"patch {pm.patch_id} layout mismatch")
                return
        self.epoch = h.epoch
        self.manifest = man
        self._reset_seqs()
        self._trackers = {pm.patch_id: SequenceTracker(self.s, self.m, self._erasure, pm.rounds + 1,
                                                       self.cfg.reorder_timeout_ps,
                                                       deliver_late=self.cfg.deliver_late,
                                                       name=f"seq.p{pm.patch_id}", patch=pm.patch_id)
                          for pm in man.patches}
        self._round_sizes = {pm.patch_id: [len(a) for a in pm.round_acqs] for pm in man.patches}
        self._credit_base = self.fifo.popped
        self._last_credit_sent = self.fifo.popped
        self._hello_ack(F.HELLO_OK)
        self._set_state(LinkState.SYNCING, "hello accepted")
        self._start_keepalive()
        if self.ts_slave.locked or not self.cfg.require_time_lock:
            self._set_state(LinkState.UP, "time already locked")
        if self.sink is not None:
            self.sink.on_session("config", {"manifest": man, "epoch": self.epoch})

    def _on_lock_change(self, locked: bool) -> None:
        if locked and self.state == LinkState.SYNCING:
            self._set_state(LinkState.UP, "time locked")
        elif not locked and self.state == LinkState.UP and self.cfg.require_time_lock:
            self._set_state(LinkState.SYNCING, "time lock lost")

    def _on_state(self, st: LinkState) -> None:
        # advertise state changes immediately (don't wait for the next heartbeat)
        if self.tx is not None and st in (LinkState.UP, LinkState.SYNCING):
            self._send(FrameType.HEARTBEAT, F.encode_heartbeat(int(st), self._credit_limit(), self.frames_tx,
                                                               self.frames_rx, self.rx_errors))
        if self.sink is not None:
            self.sink.on_session("state", {"state": st.name})

    # ------------------------------------------------------------ ingress: SYN
    def _syn_header(self, h: Header, rxf: RxFrame) -> None:
        """Cut-through processing at header time: admission + sequence accounting."""
        if self.state != LinkState.UP:
            rxf.ctx["drop"] = "not_up"
            return
        if h.epoch != self.epoch:
            rxf.ctx["drop"] = "epoch"
            self.m.counter("syn.epoch_mismatch").inc()
            return
        tr = self._trackers.get(h.patch)
        if tr is None:
            rxf.ctx["drop"] = "unknown_patch"
            self.m.counter("syn.unknown_patch").inc()
            return
        status = tr.observe(h.seq, h.shot, h.round)
        rxf.ctx["seq_status"] = status
        if status == "dup":
            rxf.ctx["drop"] = "dup"

    def _on_syn(self, h: Header, pf: ParsedFrame) -> None:
        drop = pf.rxf.ctx.get("drop")
        if drop:
            self.m.counter(f"syn.dropped_{drop}").inc()
            return
        bits, soft, leak = F.decode_syn_payload(pf.payload, bool(h.flags & FLAG_LEAK))
        sizes = self._round_sizes.get(h.patch, [])
        if h.round >= len(sizes) or len(bits) != sizes[h.round]:
            self.m.counter("syn.layout_mismatch").inc()
            self._erasure(ErasureRecord(h.patch, h.seq, "layout", h.shot, h.round, False, self.clock.now_ps()))
            return
        rec = SyndromeRecord(patch=h.patch, shot=h.shot, round=h.round, bits=bits, soft=soft, leak=leak,
                             flags=h.flags, origin_ts=h.origin_ts, seq=h.seq, sof_ts=pf.rxf.sof_ts,
                             late=pf.rxf.ctx.get("seq_status") == "late")
        if self.ts_slave.locked:
            self.m.histogram("syn.meas_to_arrival_ps").record(max(0, rec.sof_ts - rec.origin_ts))
        self.fifo.push(rec)

    def _rx_error(self, reason: str, h: Optional[Header], rxf: RxFrame) -> None:
        if h is not None and h.type == FrameType.SYN and self.state == LinkState.UP \
                and rxf.ctx.get("seq_status") in ("ok", "reordered", "late"):
            # header was good: we know exactly which round was lost
            tr = self._trackers.get(h.patch)
            if tr is not None:
                tr.mark_corrupt(h.seq)
            self.m.counter(f"syn.erasure_{reason}").inc()
            self._erasure(ErasureRecord(h.patch, h.seq, reason, h.shot, h.round, False, self.clock.now_ps()))

    def _erasure(self, e: ErasureRecord) -> None:
        self.m.counter("erasures").inc()
        if self.sink is not None:
            self.sink.on_erasure(e)

    def _fifo_overflow(self, rec: SyndromeRecord) -> None:
        self._erasure(ErasureRecord(rec.patch, rec.seq, "overflow", rec.shot, rec.round, False,
                                    self.clock.now_ps()))

    def _to_sink(self, rec: SyndromeRecord) -> None:
        rec.ready_ts = self.clock.now_ps()
        if self.ts_slave.locked:
            self.m.histogram("syn.meas_to_ready_ps").record(max(0, rec.ready_ts - rec.origin_ts))
            self.m.histogram("syn.ingress_pipeline_ps").record(max(0, rec.ready_ts - rec.sof_ts))
        if self.sink is not None:
            self.sink.on_syndrome(rec)

    # ------------------------------------------------------------ credits
    def _credit_limit(self) -> int:
        return (self.fifo.popped - self._credit_base + self.cfg.fifo_capacity) & 0xFFFFFFFF

    def _on_pop(self, popped: int) -> None:
        if popped - self._last_credit_sent >= self.cfg.credit_update_every or len(self.fifo) == 0:
            if popped != self._last_credit_sent and self.state == LinkState.UP:
                self._last_credit_sent = popped
                self._send(FrameType.CREDIT, F.encode_credit(self._credit_limit(), len(self.fifo),
                                                             1 if self.fifo.above_high else 0))

    def _on_watermark(self, high: bool) -> None:
        self.m.counter("backpressure_high" if high else "backpressure_clear").inc()
        if self.state == LinkState.UP:
            self._send(FrameType.CREDIT, F.encode_credit(self._credit_limit(), len(self.fifo), 1 if high else 0))

    # ------------------------------------------------------------ egress: corrections
    def send_correction(self, cor: CorrectionRecord) -> None:
        if self.state != LinkState.UP:
            self.m.counter("cor.dropped_link_not_up").inc()
            return
        payload = F.encode_cor_payload(cor.obs_flips, cor.status, cor.decoder_latency_ns, max(cor.ref_seq, 0))
        cor.origin_ts = self.clock.now_ps()
        cor.seq = self._send(FrameType.COR, payload, patch=cor.patch, shot=cor.shot, rnd=cor.round,
                             origin_ts=cor.origin_ts)

"""
AURA-Link over UDP/IP: the real-network transport backend.

Runs the *same* endpoint code (framing, parser, timestamp tagger, sequence
tracker, elastic FIFO, time sync, session, management ARQ) against real
sockets and the host monotonic clock, driven by a :class:`RealtimeScheduler`.
Use it to connect the virtual control system to a decoder running in another
process or on another machine, or to soak-test the protocol over a real
network.  One AURA-Link frame per datagram (frames are < 64 KiB).

Timestamps come from the OS (software timestamping), so time-sync accuracy is
limited to the host's scheduling jitter (typically 10-100 us) -- use the
serial PHY model or a NIC with hardware timestamping for nanosecond work.
"""

from __future__ import annotations

import socket
import threading
from typing import Callable, Optional

from ..core.errors import LinkError
from ..core.sim import RealtimeScheduler
from ..core.telemetry import MetricsRegistry
from .frames import HEADER_LEN
from .phy import RxFrame


class UdpTransport:
    def __init__(self, sched: RealtimeScheduler, bind: tuple[str, int], peer: Optional[tuple[str, int]] = None,
                 metrics: Optional[MetricsRegistry] = None, name: str = "udp"):
        self.s = sched
        self.m = metrics or MetricsRegistry(name)
        self.name = name
        self.peer = peer
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 << 20)
        except OSError:  # pragma: no cover
            pass
        self.sock.bind(bind)
        self.sock.settimeout(0.2)
        self.local_addr = self.sock.getsockname()
        self._rx_sof: Optional[Callable[[RxFrame], None]] = None
        self._rx_hdr: Optional[Callable[[RxFrame], None]] = None
        self._rx_eof: Optional[Callable[[RxFrame], None]] = None
        self._closed = False
        self._thread = threading.Thread(target=self._rx_loop, name=f"{name}-rx", daemon=True)

    def attach_endpoint(self, ep) -> None:
        """Wire this transport to a LinkEndpoint (both directions) and start receiving."""
        self._rx_sof, self._rx_hdr, self._rx_eof = ep.phy_sof, ep.phy_header, ep.phy_deliver
        ep.attach_tx(self)
        if not self._thread.is_alive():
            self._thread.start()

    # -- transmit (called on the scheduler thread) ------------------------
    def send(self, frame: bytes, channel_class: int = 0, on_wire_sof=None) -> None:
        if self.peer is None:
            raise LinkError(f"{self.name}: peer address unknown (no datagram received yet)")
        t = self.s.now_ps()
        try:
            self.sock.sendto(frame, self.peer)
            self.m.counter("tx.datagrams").inc()
        except OSError as e:
            self.m.counter("tx.errors").inc()
            raise LinkError(f"{self.name}: send failed: {e}") from e
        # two-step egress timestamp: report *after* the datagram is out, so any
        # follow-up message (e.g. TS_FOLLOW_UP) can never overtake it
        if on_wire_sof is not None:
            on_wire_sof(t)

    # -- receive ------------------------------------------------------------
    def _rx_loop(self) -> None:
        while not self._closed:
            try:
                data, addr = self.sock.recvfrom(65536)
            except socket.timeout:
                continue
            except OSError:
                break
            if self.peer is None:
                self.peer = addr                     # learn the peer from the first datagram
            self.m.counter("rx.datagrams").inc()
            self.s.call_soon(self._deliver, bytes(data))

    def _deliver(self, data: bytes) -> None:
        now = self.s.now_ps()
        rxf = RxFrame(data=data, error="", t_sof_pcs=now, t_wire_sof=now, block_core_ts=[now],
                      rx_latency_ps=0, nblocks=1)
        if len(data) >= HEADER_LEN:
            rxf.header_bytes = data[:HEADER_LEN]
            rxf.t_header_core = now
        if self._rx_sof:
            self._rx_sof(rxf)
        if rxf.header_bytes is not None and self._rx_hdr:
            self._rx_hdr(rxf)
        if self._rx_eof:
            self._rx_eof(rxf)

    def close(self) -> None:
        self._closed = True
        try:
            self.sock.close()
        except OSError:  # pragma: no cover
            pass
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)

"""
Real-socket AURA-Link soak test: a control endpoint and an ingress gateway in
one process, talking over UDP on localhost (or across machines), with the
real wall clock.  Streams synthetic syndrome rounds for a compiled
experiment's manifest and has an oracle decoder answer every shot.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..core.clock import LocalClock, TrueTime
from ..core.sim import RealtimeScheduler
from ..core.units import MS, US
from ..link.endpoint import ControlEndpoint, IngressGateway, LinkConfig
from ..link.records import FLAG_FIRST, FLAG_LAST, SyndromeRecord
from ..link.udp import UdpTransport
from .decoder_stubs import OracleDecoderStub


@dataclass
class LoopbackReport:
    shots: int
    frames_sent: int
    frames_delivered: int
    corrections: int
    erasures: int
    session_s: float
    run_s: float
    time_offset_us: Optional[float]
    rtt_us_p50: Optional[float]
    rtt_us_p99: Optional[float]
    ok: bool

    def summary(self) -> str:
        return (f"UDP loopback: {self.shots} shots, {self.frames_sent} SYN frames -> {self.frames_delivered} delivered, "
                f"{self.corrections} corrections, {self.erasures} erasures; session up in {self.session_s:.2f} s; "
                f"clock offset {self.time_offset_us if self.time_offset_us is None else round(self.time_offset_us, 1)} us; "
                f"SYN->COR round trip p50 {self.rtt_us_p50} us p99 {self.rtt_us_p99} us; "
                f"{'PASS' if self.ok else 'FAIL'}")


def run_udp_loopback(manifest, shots: int = 100, shot_period_s: float = 0.002, host: str = "127.0.0.1",
                     timeout_s: float = 20.0, seed: int = 0) -> LoopbackReport:
    cfg = LinkConfig.for_udp()
    sched = RealtimeScheduler("aura-udp")
    try:
        cclk = LocalClock(TrueTime(sched), name="control")
        gclk = LocalClock(TrueTime(sched), offset_ps=3 * MS, freq_ppm=30.0, name="gateway")
        ctrl = ControlEndpoint(sched, cfg, cclk, name="control-link")
        stub = OracleDecoderStub(sched, None, base_latency_ps=10 * US)
        gw = IngressGateway(sched, cfg, gclk, stub, name="gateway")
        stub.attach(gw)
        gt = UdpTransport(sched, (host, 0), name="udp-gw")
        ct = UdpTransport(sched, (host, 0), peer=gt.local_addr, name="udp-ctl")
        gt.peer = ct.local_addr
        gt.attach_endpoint(gw)
        ct.attach_endpoint(ctrl)
        ready = threading.Event()
        rtts: list[float] = []
        sent_at: dict[tuple[int, int], float] = {}
        cor_count = [0]

        def on_cor(cor) -> None:
            cor_count[0] += 1
            t = sent_at.pop((cor.patch, cor.shot), None)
            if t is not None:
                rtts.append((time.perf_counter() - t) * 1e6)
        ctrl.on_correction = on_cor
        ctrl.on_ready = ready.set
        t0 = time.perf_counter()
        sched.call_soon(ctrl.open, manifest)
        if not ready.wait(timeout_s):
            raise TimeoutError(f"AURA-Link session did not come up over UDP (state {ctrl.state.name})")
        session_s = time.perf_counter() - t0
        rng = np.random.default_rng(seed)
        frames = 0
        t1 = time.perf_counter()
        for shot in range(shots):
            for pm in manifest.patches:
                for r, accs in enumerate(pm.round_acqs):
                    bits = rng.integers(0, 2, len(accs)).astype(np.uint8)
                    flags = (FLAG_FIRST if r == 0 else 0) | (FLAG_LAST if r == pm.rounds else 0)
                    rec = SyndromeRecord(pm.patch_id, shot, r, bits, flags=flags, origin_ts=0)
                    if r == pm.rounds:
                        sent_at[(pm.patch_id, shot)] = time.perf_counter()

                    def send(rec=rec) -> None:
                        rec.origin_ts = cclk.now_ps()
                        ctrl.send_syndrome(rec)
                    sched.call_soon(send)
                    frames += 1
            time.sleep(shot_period_s)
        deadline = time.perf_counter() + timeout_s
        expected = shots * len(manifest.patches)
        while cor_count[0] < expected and time.perf_counter() < deadline:
            time.sleep(0.01)
        run_s = time.perf_counter() - t1
        delivered = gw.fifo.popped
        er = len(stub.erasures)
        off = gw.ts_slave.last_offset_ps
        rt = np.array(rtts) if rtts else None
        ok = (cor_count[0] == expected and delivered == frames and er == 0 and not sched.errors)
        rep = LoopbackReport(shots, frames, delivered, cor_count[0], er, session_s, run_s,
                             None if off is None else off / 1e6,
                             None if rt is None else round(float(np.percentile(rt, 50)), 1),
                             None if rt is None else round(float(np.percentile(rt, 99)), 1), ok)
        ctrl.close()
        gt.close()
        ct.close()
        return rep
    finally:
        sched.close()

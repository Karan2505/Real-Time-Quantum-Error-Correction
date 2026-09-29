"""
Wiring helpers: build a full-duplex AURA-Link between a control-system
endpoint and an ingress gateway over two simulated serial lanes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..core.clock import LocalClock, TrueTime
from ..core.rng import RngFactory
from ..core.telemetry import MetricsRegistry
from .endpoint import ClockConfig, ControlEndpoint, IngressGateway, LinkConfig
from .ingress import IngressSink
from .phy import Impairments, PhyRx, PhyTx, SerialChannel


@dataclass
class SimLink:
    control: ControlEndpoint
    gateway: IngressGateway
    down: SerialChannel          # control -> gateway (syndromes)
    up: SerialChannel            # gateway -> control (corrections)
    control_tx: PhyTx
    gateway_tx: PhyTx
    control_rx: PhyRx
    gateway_rx: PhyRx
    metrics: MetricsRegistry

    def set_impairments(self, down: Optional[Impairments] = None, up: Optional[Impairments] = None) -> None:
        if down is not None:
            self.down.set_impairments(down)
        if up is not None:
            self.up.set_impairments(up)


def _clock(sched, cc: ClockConfig, rng, name: str) -> LocalClock:
    return LocalClock(TrueTime(sched), offset_ps=cc.offset_ps, freq_ppm=cc.freq_ppm,
                      resolution_ps=cc.resolution_ps, jitter_ps=cc.jitter_ps,
                      wander_ppm_per_sqrt_s=cc.wander_ppm_per_sqrt_s, rng=rng, name=name)


def build_sim_link(sched, cfg: Optional[LinkConfig] = None, rng: Optional[RngFactory] = None,
                   sink: Optional[IngressSink] = None, control_clock: Optional[LocalClock] = None,
                   down_impairments: Optional[Impairments] = None,
                   up_impairments: Optional[Impairments] = None) -> SimLink:
    cfg = cfg or LinkConfig()
    cfg.validate()
    rngf = rng or RngFactory(0)
    mc = MetricsRegistry("link.control")
    mg = MetricsRegistry("link.gateway")
    cclk = control_clock or _clock(sched, cfg.control_clock, rngf.stream("link/control_clock"), "control")
    gclk = _clock(sched, cfg.gateway_clock, rngf.stream("link/gateway_clock"), "gateway")
    ctrl = ControlEndpoint(sched, cfg, cclk, mc)
    gw = IngressGateway(sched, cfg, gclk, sink, mg)
    down = SerialChannel(sched, cfg.phy, rngf.stream("link/down"), mc, down_impairments, name="lane.down")
    up = SerialChannel(sched, cfg.phy, rngf.stream("link/up"), mg, up_impairments, name="lane.up")
    ctx = PhyTx(sched, cfg.phy, down, mc, name="phy_tx")
    gtx = PhyTx(sched, cfg.phy, up, mg, name="phy_tx")
    grx = PhyRx(sched, cfg.phy, mg, gw.phy_deliver, gw.phy_sof, gw.phy_header, name="phy_rx")
    crx = PhyRx(sched, cfg.phy, mc, ctrl.phy_deliver, ctrl.phy_sof, ctrl.phy_header, name="phy_rx")
    down.rx = grx
    up.rx = crx
    ctrl.attach_tx(ctx)
    gw.attach_tx(gtx)
    return SimLink(ctrl, gw, down, up, ctx, gtx, crx, grx, MetricsRegistry("link"))

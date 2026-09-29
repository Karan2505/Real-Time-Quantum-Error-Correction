"""Helpers for AURA-Link session tests."""
import numpy as np

from aura.core import RngFactory, Simulator, US
from aura.link import LinkConfig, build_sim_link
from aura.link.records import FLAG_FIRST, FLAG_LAST, SyndromeRecord

from conftest import make_compiled


class Collector:
    def __init__(self):
        self.recs, self.eras, self.events, self.manifest = [], [], [], None

    def on_syndrome(self, r):
        self.recs.append(r)

    def on_erasure(self, e):
        self.eras.append(e)

    def on_session(self, ev, info):
        self.events.append((ev, info))
        if ev == "config":
            self.manifest = info["manifest"]


def up_link(cfg=None, seed=0, down=None, up=None, spec="surface:3", rounds=3, provision=False, run_us=400):
    _, _, ce, _ = make_compiled(spec, rounds, "Z")
    sim = Simulator()
    col = Collector()
    link = build_sim_link(sim, cfg or LinkConfig(), RngFactory(seed), sink=col, down_impairments=down,
                          up_impairments=up)
    if provision:
        link.gateway.provision(ce.manifest)
    link.control.open(ce.manifest)
    sim.run(until_ps=run_us * US)
    return sim, link, col, ce


def stream_rounds(sim, link, ce, shots, spacing_ps=1 * US, soft=True, seed=0):
    rng = np.random.default_rng(seed)
    pm = ce.manifest.patches[0]
    sent = []
    t = sim.now + spacing_ps
    for shot in range(shots):
        for r, accs in enumerate(pm.round_acqs):
            n = len(accs)
            rec = SyndromeRecord(pm.patch_id, shot, r, rng.integers(0, 2, n).astype(np.uint8),
                                 soft=rng.integers(-7, 8, n).astype(np.int8) if soft else None,
                                 leak=rng.random(n) < 0.05,
                                 flags=(FLAG_FIRST if r == 0 else 0) | (FLAG_LAST if r == pm.rounds else 0) | 0x03)

            def send(rec=rec):
                rec.origin_ts = link.control.clock.now_ps()
                link.control.send_syndrome(rec)
            sim.schedule_at(t, send)
            sent.append(rec)
            t += spacing_ps
    return sent

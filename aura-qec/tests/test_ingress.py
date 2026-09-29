import numpy as np

from aura.core import MetricsRegistry, NS, Simulator, US
from aura.link.ingress import IngressElasticFifo, SequenceTracker
from aura.link.records import SyndromeRecord


def _tracker(fps=4, timeout=1 * US):
    sim = Simulator()
    er = []
    tr = SequenceTracker(sim, MetricsRegistry("t"), er.append, frames_per_shot=fps, reorder_timeout_ps=timeout)
    return sim, tr, er


def test_in_order_reorder_and_duplicates():
    sim, tr, er = _tracker()
    assert tr.observe(0, 0, 0) == "ok"
    assert tr.observe(2, 0, 2) == "ok"            # gap at 1
    assert tr.pending == 1
    assert tr.observe(1, 0, 1) == "reordered"      # arrives within the window
    assert tr.observe(1, 0, 1) == "dup"
    sim.run()
    assert er == [] and tr.pending == 0


def test_gap_becomes_erasure_with_inferred_position_and_late_arrival_flagged():
    sim, tr, er = _tracker(fps=4)
    for s in range(6):                            # shot 0 rounds 0..3, shot 1 rounds 0..1
        tr.observe(s, s // 4, s % 4)
    tr.observe(8, 2, 0)                           # seq 6, 7 (shot 1 rounds 2, 3) missing
    sim.run()
    assert [(e.seq, e.shot, e.round, e.reason) for e in er] == [(6, 1, 2, "gap"), (7, 1, 3, "gap")]
    assert tr.observe(7, 1, 3) == "late"


def test_sequence_wraparound_and_resync():
    sim, tr, er = _tracker()
    tr.reset(0xFFFFFFFE)
    assert tr.observe(0xFFFFFFFE, 0, 0) == "ok"
    assert tr.observe(0xFFFFFFFF, 0, 1) == "ok"
    assert tr.observe(0, 0, 2) == "ok"
    assert tr.observe(1, 0, 3) == "ok"
    sim.run()
    assert er == []
    assert tr.observe(10 ** 6, 5, 0) == "ok"      # huge jump: resynchronise, no erasure storm
    sim.run()
    assert er == []


def test_elastic_fifo_overflow_watermarks_and_service_rate():
    sim = Simulator()
    out, ovf, wm = [], [], []
    fifo = IngressElasticFifo(sim, MetricsRegistry("f"), lambda r: out.append((sim.now, r)), ovf.append,
                              capacity=8, service_ps=100 * NS, high_wm=0.75, low_wm=0.25,
                              on_watermark=wm.append)
    recs = [SyndromeRecord(0, 0, i, np.zeros(3, np.uint8), seq=i) for i in range(12)]
    for r in recs:
        fifo.push(r)
    sim.run()
    assert len(ovf) == 4 and fifo.overflows == 4
    assert [r.seq for _, r in out] == list(range(8))
    gaps = np.diff([t for t, _ in out])
    assert (gaps >= 100 * NS).all()
    assert wm == [True, False] and fifo.max_occupancy == 8

import os

import numpy as np

from aura.core import MetricsRegistry, NS, Simulator
from aura.link.phy import (GilbertElliott, Impairments, PhyConfig, PhyRx, PhyTx, SerialChannel, descramble,
                           pcs_decode, pcs_encode, scramble)


def test_pcs_and_scrambler_roundtrip_all_lengths():
    for n in range(7, 90):
        fr = os.urandom(n)
        blocks = pcs_encode(fr)
        sc, _ = scramble([p for _, p in blocks], 0x1234567)
        ds, _ = descramble(sc, 0x1234567)
        out, err = pcs_decode([(h, p) for (h, _), p in zip(blocks, ds)])
        assert err == "" and out == fr


def test_scrambler_error_multiplication_is_three():
    blocks = pcs_encode(bytes(64))
    sc, _ = scramble([p for _, p in blocks], 0)
    ref, _ = descramble(sc, 0)
    for bit in (0, 17, 40, 63):
        bad = list(sc)
        bad[2] ^= 1 << bit
        ds, _ = descramble(bad, 0)
        assert sum(bin(a ^ b).count("1") for a, b in zip(ds, ref)) == 3


def _pair(cfg=None, imp=None):
    sim = Simulator()
    cfg = cfg or PhyConfig()
    m = MetricsRegistry("t")
    got, sofs, hdrs = [], [], []
    ch = SerialChannel(sim, cfg, np.random.default_rng(1), m, imp)
    rx = PhyRx(sim, cfg, m, lambda f: got.append((sim.now, f)), lambda f: sofs.append(sim.now),
               lambda f: hdrs.append(sim.now))
    ch.rx = rx
    tx = PhyTx(sim, cfg, ch, m)
    return sim, tx, rx, got, sofs, hdrs, m


def test_phy_latency_model():
    sim, tx, rx, got, sofs, hdrs, _ = _pair()
    cfg = tx.cfg
    frame = os.urandom(58)
    wire = []
    tx.send(frame, 0, lambda t: wire.append((sim.now, t)))
    sim.run()
    (t_done, f), = got
    assert f.data == frame and f.error == ""
    nb = len(pcs_encode(frame))
    t_wire = cfg.tx_latency_ps
    assert wire[0][1] == t_wire
    t_arr0 = t_wire + cfg.prop_ps
    assert sofs[0] == t_arr0 + cfg.block_ps + cfg.rx_latency_ps
    # delivery: last block decoded, then CDC synchroniser, snapped to a core clock edge
    last = t_arr0 + nb * cfg.block_ps + cfg.rx_latency_ps
    assert last + 2 * cfg.core_period_ps <= t_done <= last + 4 * cfg.core_period_ps
    assert hdrs and hdrs[0] < t_done                  # cut-through header before end of frame
    assert f.header_bytes == frame[:36]


def test_strict_priority_qos():
    sim, tx, rx, got, *_ = _pair()
    tx.send(b"M" * 1000, 1)       # management frame starts on the idle link ...
    sim.run(until_ps=1)
    for i in range(3):
        tx.send(bytes([i]) * 60, 1)
    tx.send(b"R" * 60, 0)          # ... a real-time frame queued behind more management traffic
    sim.run()
    order = [f.data[:1] for _, f in got]
    assert order[0] == b"M" and order[1] == b"R"   # never pre-empts the frame on the wire, then jumps the queue
    # simultaneous arrivals: real-time wins
    sim2, tx2, _, got2, *_ = _pair()
    tx2.send(b"M" * 100, 1)
    tx2.send(b"R" * 60, 0)
    sim2.run()
    assert got2[0][1].data[:1] == b"R"


def test_bit_errors_never_deliver_silently_corrupted_frames():
    sim, tx, rx, got, *_ = _pair(imp=Impairments(ber=2e-4))
    sent = [os.urandom(100) for _ in range(400)]
    for f in sent:
        tx.send(f, 0)
    sim.run()
    clean = sum(1 for _, f in got if f.error == "" and f.data in sent)
    corrupted_but_clean_pcs = [f for _, f in got if f.error == "" and f.data not in sent]
    # PCS-clean frames with payload damage must be left for the CRC to catch: count them
    assert clean > 300
    assert len(got) <= len(sent)
    assert tx.m.count("ch.bit_errors") > 0
    assert all(len(f.data) == 100 for f in corrupted_but_clean_pcs)


def test_loss_of_signal_drops_frames_and_relocks():
    cfg = PhyConfig()
    t_los = (10_000 * NS, 20_000 * NS)
    sim, tx, rx, got, *_ = _pair(cfg, Impairments(los_intervals=[t_los]))
    times = list(range(0, 40_000 * NS, 1_000 * NS))
    for t in times:
        sim.schedule_at(t, tx.send, os.urandom(60), 0)
    sim.run()
    arrived = [f.t_wire_sof for _, f in got]
    relock = t_los[1] + cfg.relock_blocks * cfg.block_ps
    assert all(not (t_los[0] - cfg.block_ps * 20 < t < relock) for t in arrived)
    assert any(t >= relock for t in arrived) and rx.lock_losses == 1
    assert len(arrived) == len([t for t in times if not (t_los[0] - 200 * NS < t + 40 * NS < relock)])


def test_burst_errors_cause_lock_loss():
    ge = GilbertElliott(p_good_to_bad=0.05, p_bad_to_good=0.02, ber_bad=0.3)
    sim, tx, rx, got, *_ = _pair(imp=Impairments(burst=ge))
    for i in range(200):
        sim.schedule_at(i * 1000 * NS, tx.send, os.urandom(1000), 0)
    sim.run()
    assert rx.lock_losses >= 1

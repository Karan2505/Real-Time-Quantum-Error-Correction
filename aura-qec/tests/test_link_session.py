import numpy as np
import pytest

from aura.core import NS, US
from aura.link import GilbertElliott, Impairments, LinkConfig, LinkState

from linkutil import stream_rounds, up_link


def test_session_bring_up_with_config_push_and_time_lock():
    sim, link, col, ce = up_link()
    c, g = link.control, link.gateway
    assert c.state == LinkState.UP and g.state == LinkState.UP
    assert g.m.count("config_received") == 1 and col.manifest is not None
    assert col.manifest.config_epoch() == ce.config_epoch
    # gateway clock started 37 us off and 12.5 ppm fast; now synchronised to the control clock
    assert abs(g.clock.now_ps() - c.clock.now_ps()) < 2 * NS
    cfg = link.control.cfg.phy
    assert g.ts_slave.last_delay_ps == pytest.approx(cfg.prop_ps, abs=2 * NS)
    names = [s.split()[0] for _, s in g.state_log]
    assert names == ["DOWN->SYNCING", "SYNCING->UP"]


def test_provisioned_gateway_skips_config_push():
    sim, link, col, ce = up_link(provision=True)
    assert link.gateway.m.count("config_received") == 0
    assert link.control.state == LinkState.UP


def test_syndrome_frames_delivered_intact_in_order_with_timestamps():
    sim, link, col, ce = up_link()
    sent = stream_rounds(sim, link, ce, shots=30)
    sim.run(until_ps=sim.now + 300 * US)
    assert len(col.recs) == len(sent) and not col.eras
    for s, r in zip(sent, col.recs):
        assert (r.patch, r.shot, r.round, r.seq) == (s.patch, s.shot, s.round, s.seq)
        assert np.array_equal(r.bits, s.bits) and np.array_equal(r.soft, s.soft) and np.array_equal(r.leak, s.leak)
        assert 0 < r.sof_ts - r.origin_ts < 200 * NS          # synced clocks: true one-way latency
        assert r.ready_ts > r.sof_ts and not r.late


@pytest.mark.parametrize("ber", [3e-5, 2e-4])
def test_every_frame_is_delivered_or_erased_under_bit_errors(ber):
    sim, link, col, ce = up_link()
    link.set_impairments(down=Impairments(ber=ber))
    sent = stream_rounds(sim, link, ce, shots=150)
    sim.run(until_ps=sim.now + 2000 * US)
    got = {r.seq for r in col.recs}
    erased = {e.seq for e in col.eras}
    assert got | erased == {s.seq for s in sent}
    assert not (got & erased) or all(r.late for r in col.recs if r.seq in erased)
    # erasures carry the lost round's position (exact for payload CRC errors, inferred for gaps)
    by_seq = {s.seq: s for s in sent}
    for e in col.eras:
        assert (e.shot, e.round) == (by_seq[e.seq].shot, by_seq[e.seq].round)
    assert col.eras, "expected some erasures at this BER"
    delivered = {r.seq: r for r in col.recs}
    for seq, r in delivered.items():                          # nothing corrupted slips through
        assert np.array_equal(r.bits, by_seq[seq].bits)


def test_mgmt_config_transfer_survives_a_lossy_link():
    ge = GilbertElliott(p_good_to_bad=0.002, p_bad_to_good=0.2, ber_bad=0.01)
    sim, link, col, ce = up_link(down=Impairments(burst=ge), run_us=4000)
    assert link.control.state == LinkState.UP
    assert link.control.m.count("mgmt.timeouts") >= 0
    assert col.manifest.config_epoch() == ce.config_epoch


def test_liveness_timeout_on_loss_of_signal_and_automatic_recovery():
    cfg = LinkConfig()
    sim, link, col, ce = up_link(cfg)
    t0 = sim.now
    both = [(t0, t0 + 1000 * US)]
    link.set_impairments(down=Impairments(los_intervals=both), up=Impairments(los_intervals=both))
    sim.run(until_ps=t0 + 800 * US)
    assert link.gateway.state == LinkState.DOWN and link.control.state != LinkState.UP
    sim.run(until_ps=t0 + 1600 * US)                      # no manual intervention
    assert link.control.state == LinkState.UP and link.gateway.state == LinkState.UP
    assert link.control.m.count("reconnects") >= 1


def test_half_open_session_is_detected_and_repaired():
    """Only the control->gateway lane fails: the gateway times out while the control
    still hears the gateway.  The gateway must force a re-handshake (CLOSE)."""
    sim, link, col, ce = up_link()
    t0 = sim.now
    link.set_impairments(down=Impairments(los_intervals=[(t0, t0 + 400 * US)]))
    sim.run(until_ps=t0 + 390 * US)
    assert link.gateway.state == LinkState.DOWN
    sim.run(until_ps=t0 + 900 * US)
    assert link.gateway.m.count("close_no_session") >= 1
    assert link.control.state == LinkState.UP and link.gateway.state == LinkState.UP
    sent = stream_rounds(sim, link, ce, shots=5)          # traffic flows again
    sim.run(until_ps=sim.now + 100 * US)
    assert {r.seq for r in col.recs[-len(sent):]} == {s.seq for s in sent}


def test_credit_based_backpressure_protects_a_slow_consumer():
    cfg = LinkConfig(fifo_capacity=16, fifo_service_ps=2 * US, credit_update_every=2)
    sim, link, col, ce = up_link(cfg)
    c = link.control
    assert c.credits_available == 16
    fps = c.frames_per_shot
    shots_started = 0
    for _ in range(40):
        if c.can_start_shot():
            shots_started += 1
            stream_rounds(sim, link, ce, shots=1, spacing_ps=50 * NS, seed=shots_started)
        sim.run(until_ps=sim.now + 1 * US)
    sim.run(until_ps=sim.now + 200 * US)
    assert link.gateway.fifo.overflows == 0
    assert 0 < shots_started < 40                              # throttled at shot boundaries
    assert len(col.recs) == shots_started * fps


def test_frames_with_a_foreign_epoch_are_dropped():
    sim, link, col, ce = up_link()
    link.control.epoch ^= 0x5A5A5A5A
    stream_rounds(sim, link, ce, shots=2)
    sim.run(until_ps=sim.now + 50 * US)
    assert not col.recs and link.gateway.m.count("syn.dropped_epoch") > 0


def test_corrections_flow_back_to_the_control_system():
    sim, link, col, ce = up_link()
    got = []
    link.control.on_correction = got.append
    from aura.link.records import CorrectionRecord
    for shot in range(5):
        sim.schedule(shot * US, link.gateway.send_correction,
                     CorrectionRecord(0, shot, 3, np.array([shot % 2], np.uint8), status=1, ref_seq=shot))
    sim.run(until_ps=sim.now + 20 * US)
    assert [(g.shot, int(g.obs_flips[0])) for g in got] == [(s, s % 2) for s in range(5)]
    assert all(0 < g.arrival_ts - g.origin_ts < 200 * NS for g in got)

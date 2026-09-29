import json
import time
import urllib.request

import numpy as np
import pytest

from aura.core import US
from aura.lab import LabConfig, PatchConfig, VirtualLab
from aura.link import Impairments


@pytest.fixture(scope="module")
def lab():
    lab = VirtualLab(LabConfig(seed=11, patches=[PatchConfig(0, "surface:3")], cosmic_rays=False))
    lab.calibrate("oracle")
    return lab


def test_oracle_decoder_closes_the_loop_exactly(lab):
    r = lab.run(lab.experiment(rounds=3), shots=80, decoder="oracle", mode="stream")
    p = r.patches[0]
    assert p.missing_corrections == 0 and p.logical_errors == 0
    lat = r.latency["measurement_to_correction_ps"]
    assert lat["count"] == 80 and lat["max"] < 3 * US
    tm = lab.control.timing
    floor = tm.readout_chain_ps + tm.classify_ps + tm.crossbar_ps + tm.packetize_ps
    assert r.latency["meas_to_arrival_ps"]["min"] > floor


def test_matching_decoder_reduces_logical_errors(lab):
    r = lab.run(lab.experiment(rounds=3), shots=4000, decoder="matching", mode="batch")
    p = r.patches[0]
    assert p.logical_errors < 0.5 * p.raw_logical_errors
    assert abs(p.detection_rate - p.predicted_detection_rate) < 0.15 * p.predicted_detection_rate


def test_stream_and_batch_modes_agree_statistically(lab):
    exp = lab.experiment(rounds=3)
    a = lab.run(exp, shots=400, decoder="null", mode="stream").patches[0]
    b = lab.run(exp, shots=4000, decoder="null", mode="batch").patches[0]
    se = np.sqrt(b.detection_rate * (1 - b.detection_rate) / (400 * 24)) * 2.5
    assert abs(a.detection_rate - b.detection_rate) < 4 * se


def test_feed_forward_branching(lab):
    r = lab.run(lab.experiment(rounds=3, branching=True), shots=40, decoder="matching", mode="stream")
    br = r.branching
    assert br["timeouts"] == 0 and br["flag_matches_decision"] == 1.0
    assert br["response_ps"]["count"] == 40 and br["response_ps"]["max"] < 5 * US


def test_link_errors_become_erasures_not_silent_corruption(lab):
    lab.link.set_impairments(down=Impairments(ber=1e-4))
    try:
        r = lab.run(lab.experiment(rounds=3), shots=150, decoder="oracle", mode="stream")
    finally:
        lab.link.set_impairments(down=Impairments())
    er = sum(v for k, v in r.decoder_stats["counters"].items() if k.startswith("erasure."))
    assert er > 0
    p = r.patches[0]
    # shots whose final frame survived are still perfectly corrected by the oracle
    assert p.logical_errors <= p.missing_corrections + er


def test_two_patch_lab_streams_both_patches():
    lab2 = VirtualLab(LabConfig(seed=5, patches=[PatchConfig(0, "surface:3"), PatchConfig(3, "rep:5")],
                                cosmic_rays=False))
    lab2.calibrate("oracle")
    r = lab2.run(lab2.experiment(rounds=2), shots=30, decoder="oracle", mode="stream")
    assert [p.patch_id for p in r.patches] == [0, 3]
    assert all(p.logical_errors == 0 and p.missing_corrections == 0 for p in r.patches)


def test_lab_config_file_loads():
    lab3 = VirtualLab.from_file("configs/lab_two_patches.toml")
    assert sorted(lab3.codes) == [0, 1] and lab3.link_cfg.phy.line_rate_gbps == 25.78125


def test_console_api(lab):
    from aura.host.console import ConsoleServer
    srv = ConsoleServer(lab, "127.0.0.1", 0, token="t0k").start()
    base = f"http://127.0.0.1:{srv.address[1]}"

    def call(path, body=None, token="t0k"):
        req = urllib.request.Request(base + path, data=None if body is None else json.dumps(body).encode(),
                                     headers={"X-Aura-Token": token, "Content-Type": "application/json"},
                                     method="GET" if body is None else "POST")
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, json.loads(r.read())
    try:
        with pytest.raises(urllib.error.HTTPError):
            call("/api/status", token="wrong")
        st, s = call("/api/status")
        assert st == 200 and s["device"]["qubits"] == lab.spec.num_qubits
        with pytest.raises(urllib.error.HTTPError):
            call("/api/run", {"rounds": 0})
        st, j = call("/api/run", {"rounds": 2, "shots": 20, "decoder": "oracle", "mode": "stream"})
        assert st == 202
        for _ in range(300):
            _, job = call(f"/api/jobs/{j['job']}")
            if job["state"] in ("done", "failed"):
                break
            time.sleep(0.05)
        assert job["state"] == "done", job
        rid = job["result"]["run_id"]
        _, run = call(f"/api/runs/{rid}")
        assert run["patches"][0]["logical_errors"] == 0
        html = urllib.request.urlopen(urllib.request.Request(base + "/", headers={"X-Aura-Token": "t0k"})).read()
        assert b"AURA-QEC" in html
    finally:
        srv.stop()


def test_udp_loopback_over_real_sockets():
    from conftest import make_compiled
    from aura.testbed.udp_loopback import run_udp_loopback
    _, _, ce, _ = make_compiled("surface:3", 2, "Z")
    rep = run_udp_loopback(ce.manifest, shots=60, shot_period_s=0.001)
    assert rep.ok, rep.summary()

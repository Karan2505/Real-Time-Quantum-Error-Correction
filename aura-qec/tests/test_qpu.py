import dataclasses

import numpy as np
import pytest

from aura.codes import get_code
from aura.core import HardwareFault, RngFactory, Simulator, TimingViolation
from aura.host.calibration import CalibrationStore
from aura.host.compiler import QecCompiler, memory_experiment
from aura.qpu import PROFILES, VirtualQPU, build_device
from aura.qpu.device import DriftParams
from aura.qpu.drift import DriftModel
from aura.qpu.program import QpuProgram, StaticOp
from aura.qpu.readout import sample_iq

stim = pytest.importorskip("stim")


def _setup(spec="surface:3", basis="Z", profile="nisq", leak=False, rounds=3):
    prof = PROFILES[profile]
    if not leak:
        prof = dataclasses.replace(prof, name=profile + "_noleak", leak_cz=0.0, leak_sq=0.0)
    code = get_code(spec)
    dev = build_device([(0, code)], prof, seed=5)
    qpu = VirtualQPU(dev, RngFactory(9))
    qpu.upload_settings(qpu.oracle_control_settings(0))
    snap = qpu.snapshot(0)
    ce = QecCompiler(dev.spec, CalibrationStore(dev.spec)).compile(
        memory_experiment(code, rounds, basis, allow_uncalibrated=True), snapshot=snap, meas_flip=[0.0] * dev.n)
    qpu.load_program(ce.qpu_program)
    return code, dev, qpu, ce, snap


@pytest.mark.parametrize("spec,basis", [("surface:3", "Z"), ("surface:3", "X"), ("rep:5", "Z")])
def test_qpu_sampling_matches_its_detector_error_model(spec, basis):
    """Detection-event rates of the virtual QPU (projected bits, before readout)
    must agree with the DEM built from the same true noise parameters."""
    _, dev, qpu, ce, _ = _setup(spec, basis)
    pm = ce.manifest.patches[0]
    shots = 20000
    out = qpu.run_batch(shots, t_start_ps=0)
    ev = pm.detection_events(out["true_bits"])
    rate = ev.mean(0)
    pred = pm.dem.detector_marginals()
    z = (rate - pred) / np.sqrt(pred * (1 - pred) / shots)
    assert np.abs(z).max() < 4.5, np.abs(z).max()
    # detection events are correlated: use the empirical standard error of the per-shot mean
    se = ev.mean(1).std() / np.sqrt(shots)
    assert abs(rate.mean() - pred.mean()) < 4 * se, (rate.mean(), pred.mean(), se)


def test_noiseless_device_produces_no_detection_events():
    code = get_code("surface:3")
    dev = build_device([(0, code)], "ideal", seed=1)
    qpu = VirtualQPU(dev, RngFactory(1))
    qpu.upload_settings(qpu.oracle_control_settings(0))
    ce = QecCompiler(dev.spec, CalibrationStore(dev.spec)).compile(
        memory_experiment(code, 3, allow_uncalibrated=True, build_dem=False))
    qpu.load_program(ce.qpu_program)
    out = qpu.run_batch(200, t_start_ps=0)
    pm = ce.manifest.patches[0]
    assert not pm.detection_events(out["true_bits"]).any()
    assert not pm.observable_flips(out["true_bits"]).any()


def test_miscalibration_increases_errors():
    _, dev, qpu, ce, _ = _setup("surface:3", "Z", profile="sota")
    pm = ce.manifest.patches[0]
    good = pm.detection_events(qpu.run_batch(4000, t_start_ps=0)["true_bits"]).mean()
    bad = qpu.oracle_control_settings(0)
    bad.sx_amp = bad.sx_amp * 1.05
    bad.x180_amp = bad.x180_amp * 1.05
    bad.drive_freq_hz = bad.drive_freq_hz + 200e3
    bad.version += 1
    qpu.upload_settings(bad)
    worse = pm.detection_events(qpu.run_batch(4000, t_start_ps=0)["true_bits"]).mean()
    assert worse > good * 1.5


def test_readout_physics_assignment_errors():
    rng = np.random.default_rng(0)
    from aura.qpu.device import ReadoutTruth
    ro = ReadoutTruth(0j, 2 + 0j, 3 + 1j, 0.3, 0.0)
    n = 200000
    iq1, post1 = sample_iq(np.ones(n, int), ro, 50e6, 500e3, 0.0, rng)
    # decays during the window leave the qubit in |0>
    assert abs((post1 == 0).mean() - (1 - np.exp(-500e3 / 50e6))) < 2e-3
    bit = np.abs(iq1 - 2) < np.abs(iq1 - 0)
    p10 = 1 - bit.mean()
    expected = 1 - np.exp(-250e3 / 50e6)          # decay in the first half of the window
    assert abs(p10 - expected) < 1.5e-3
    iq2, _ = sample_iq(np.full(1000, 2), ro, 1e30, 500e3, 0.0, rng)
    assert np.mean(np.abs(iq2 - (3 + 1j)) < 1.0) > 0.99


def test_leakage_is_visible_in_readout():
    prof = dataclasses.replace(PROFILES["nisq"], name="leaky", leak_cz=0.02)
    code = get_code("surface:3")
    dev = build_device([(0, code)], prof, seed=2)
    qpu = VirtualQPU(dev, RngFactory(2))
    qpu.upload_settings(qpu.oracle_control_settings(0))
    ce = QecCompiler(dev.spec, CalibrationStore(dev.spec)).compile(
        memory_experiment(code, 3, allow_uncalibrated=True, build_dem=False))
    qpu.load_program(ce.qpu_program)
    out = qpu.run_batch(2000, t_start_ps=0)
    assert out["leaked"].mean() > 0.005


def _tiny_program(ops):
    return QpuProgram(num_qubits=2, ops=tuple(ops), num_acq=sum(o.kind == "MEASURE" for o in ops),
                      shot_duration_ps=10 ** 6)


def test_timing_and_order_violations_are_detected():
    code = get_code("rep:2")
    dev = build_device([(0, code)], "ideal", seed=1)
    sim = Simulator()
    qpu = VirtualQPU(dev, RngFactory(1), scheduler=sim)
    ops = [StaticOp(0, "SX", (0,), 24000, 0), StaticOp(1, "SX", (0,), 24000, 24000),
           StaticOp(2, "MEASURE", (0,), 500000, 48000, acq=0)]
    prog = QpuProgram(num_qubits=dev.n, ops=tuple(ops), num_acq=1, shot_duration_ps=548000)
    qpu.load_program(prog)
    qpu.begin_shot(0)
    qpu.execute([ops[0]])
    with pytest.raises(TimingViolation):
        qpu.execute([ops[1]])                      # qubit 0 still busy at t=0
    sim.run(until_ps=30000)
    with pytest.raises(HardwareFault):
        qpu.execute([ops[2]])                      # skips op 1 (not conditional)


def test_drift_is_reproducible_and_cosmic_rays_suppress_t1():
    code = get_code("surface:3")
    dp = DriftParams(cosmic_rate_per_s=2000.0, speedup=1.0)
    dev = build_device([(0, code)], "sota", seed=1, drift=dp)
    a = DriftModel(dev, np.random.default_rng(3))
    b = DriftModel(dev, np.random.default_rng(3))
    sa, sb = a.state(10 ** 10), b.state(10 ** 10)
    assert np.array_equal(sa.t1_factor, sb.t1_factor)
    assert a.cosmic_history, "expected cosmic events at 2000/s over 10 ms"
    t0, center = a.cosmic_history[-1]
    st = a.state(int(t0 * 1e12) + 1)
    assert st.t1_factor[center] < 0.2

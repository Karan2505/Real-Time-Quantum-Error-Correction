import numpy as np
import pytest

from aura.lab import LabConfig, PatchConfig, VirtualLab


@pytest.fixture(scope="module")
def calibrated_lab():
    lab = VirtualLab(LabConfig(seed=4, patches=[PatchConfig(0, "surface:3")], cosmic_rays=False))
    rep = lab.calibrate("full")
    return lab, rep


@pytest.mark.slow
def test_full_calibration_recovers_hidden_parameters(calibrated_lab):
    lab, rep = calibrated_lab
    truth = lab.qpu.oracle_control_settings(lab.sim.now)
    st = lab.store
    pi = np.array([q.pi_amp for q in st.qubits])
    assert np.max(np.abs(pi / truth.x180_amp - 1)) < 5e-3
    f = np.array([q.drive_freq_hz for q in st.qubits])
    assert np.max(np.abs(f - truth.drive_freq_hz)) < 40e3        # includes real drift during calibration
    cz = np.array([c.cz_amp for c in st.couplers])
    assert np.max(np.abs(cz / truth.cz_amp - 1)) < 5e-3
    corr = np.array([[c.corr0, c.corr1] for c in st.couplers])
    resid = (corr - truth.cz_corr + np.pi) % (2 * np.pi) - np.pi
    assert np.max(np.abs(resid)) < 0.03
    t1_true = np.array([lab.qpu.snapshot(lab.sim.now).qubits[q].t1_ps for q in range(lab.spec.num_qubits)])
    t1 = np.array([q.t1_ps for q in st.qubits])
    assert np.median(np.abs(t1 / t1_true - 1)) < 0.15
    cz_err = np.array([c.cz_error for c in st.couplers])
    cz_true = np.array([c.cz_depol for c in lab.device.couplers])
    assert np.median(np.abs(cz_err - cz_true)) < 1.5e-3
    assert st.fully_calibrated()
    assert rep["virtual_ms"] > 1000                              # calibration takes lab time


@pytest.mark.slow
def test_maintenance_does_less_work_than_bring_up(calibrated_lab):
    lab, full = calibrated_lab
    rep = lab.calibrate("maintain")
    assert "check" in rep["actions"]
    assert rep["virtual_ms"] < 0.7 * full["virtual_ms"]


@pytest.mark.slow
def test_calibrated_device_beats_uncalibrated(calibrated_lab):
    lab, _ = calibrated_lab
    exp = lab.experiment(rounds=3)
    good = lab.run(exp, shots=3000, decoder="matching", mode="batch").patches[0]
    # sabotage: revert pulse amplitudes and frequencies to fab design values
    for q, qc in enumerate(lab.store.qubits):
        lab.store.update_qubit(q, lab.sim.now, "sabotage", pi_amp=lab.spec.design_pi_amp[q],
                               sx_amp=lab.spec.design_pi_amp[q] / 2, drive_freq_hz=lab.spec.design_f01_hz[q])
    bad = lab.run(exp, shots=3000, decoder="matching", mode="batch").patches[0]
    assert bad.detection_rate > 1.5 * good.detection_rate

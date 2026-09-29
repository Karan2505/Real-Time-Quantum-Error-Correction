"""
QEC compiler facade:  ExperimentSpec + DeviceSpec + CalibrationStore  ->
CompiledExperiment (QPU program, sequencer programs, control settings,
readout discriminators, decoder manifest with DEM, config epoch).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ...control.isa import SequencerBundle
from ...control.readout import Discriminator
from ...core.errors import CompileError
from ...core.rng import RngFactory
from ...qpu import noise as N
from ...qpu.device import DeviceSpec
from ...qpu.program import QpuProgram
from ...qpu.qpu import ControlSettings
from ...qpu.reference import acq_bits, reference_sample, z_eigen_annotations
from ..calibration.store import CalibrationStore
from .builder import build_logical_circuit
from .dem import generate_dem
from .experiment import ExperimentSpec
from .lowering import LoweredProgram, lower
from .manifest import DecoderManifest, PatchManifest
from .noise_model import estimated_snapshot


@dataclass
class CompiledExperiment:
    spec: ExperimentSpec
    device: DeviceSpec
    qpu_program: QpuProgram
    sequencers: SequencerBundle
    settings: ControlSettings
    discriminators: list[Discriminator]
    manifest: DecoderManifest
    lowered: LoweredProgram
    noise_snapshot: Optional[N.NoiseSnapshot]
    z_states: list
    calibration_version: int
    config_epoch: int
    stats: dict = field(default_factory=dict)

    @property
    def num_acq(self) -> int:
        return self.qpu_program.num_acq


def build_detectors(lowered: LoweredProgram, spec: ExperimentSpec, device: DeviceSpec,
                    ref_acq: np.ndarray, flag_acq: int) -> DecoderManifest:
    acq_of = {key: a for a, key in enumerate(lowered.acq_keys)}
    patches: list[PatchManifest] = []
    acq_map = [[-1, -1, -1] for _ in range(len(lowered.acq_keys))]
    for p in spec.patches:
        code = p.code
        pid, R = p.patch_id, p.rounds
        round_acqs, round_kinds = [], []
        for r in range(R):
            accs = [acq_of[(pid, r, "x", c)] for c in range(code.num_x_checks)]
            accs += [acq_of[(pid, r, "z", c)] for c in range(code.num_z_checks)]
            round_acqs.append(accs)
            round_kinds.append(["x"] * code.num_x_checks + ["z"] * code.num_z_checks)
        round_acqs.append([acq_of[(pid, R, "d", i)] for i in range(code.n)])
        round_kinds.append(["d"] * code.n)
        for r, accs in enumerate(round_acqs):
            for slot, a in enumerate(accs):
                acq_map[a] = [pid, r, slot]
        dets, coords, dbasis = [], [], []
        xc = code.x_check_coords if code.x_check_coords is not None else np.zeros((code.num_x_checks, 2))
        zc = code.z_check_coords if code.z_check_coords is not None else np.zeros((code.num_z_checks, 2))
        first = ("z", code.hz, zc) if p.basis == "Z" else ("x", code.hx, xc)
        for c in range(first[1].shape[0]):
            dets.append([acq_of[(pid, 0, first[0], c)]])
            coords.append([float(first[2][c][0]), float(first[2][c][1]), 0.0])
            dbasis.append(first[0].upper())
        for r in range(1, R):
            for role, h, cc in (("x", code.hx, xc), ("z", code.hz, zc)):
                for c in range(h.shape[0]):
                    dets.append([acq_of[(pid, r, role, c)], acq_of[(pid, r - 1, role, c)]])
                    coords.append([float(cc[c][0]), float(cc[c][1]), float(r)])
                    dbasis.append(role.upper())
        role, h, cc = first
        for c in range(h.shape[0]):
            support = [acq_of[(pid, R, "d", int(q))] for q in np.nonzero(h[c])[0]]
            dets.append(support + [acq_of[(pid, R - 1, role, c)]])
            coords.append([float(cc[c][0]), float(cc[c][1]), float(R)])
            dbasis.append(role.upper())
        logicals = code.logical_z if p.basis == "Z" else code.logical_x
        obs = [[acq_of[(pid, R, "d", int(q))] for q in np.nonzero(l)[0]] for l in logicals]
        det_ref = [int(np.bitwise_xor.reduce(ref_acq[d].astype(np.uint8))) for d in dets]
        obs_ref = [int(np.bitwise_xor.reduce(ref_acq[o].astype(np.uint8))) for o in obs]
        patches.append(PatchManifest(
            patch_id=pid, code_name=code.name, code_fingerprint=code.fingerprint(), n=code.n, k=code.k,
            basis=p.basis, rounds=R, hx=code.hx.tolist(), hz=code.hz.tolist(), round_acqs=round_acqs,
            round_kinds=round_kinds, detectors=dets, det_coords=coords, det_basis=dbasis,
            observables=obs, det_ref=det_ref, obs_ref=obs_ref,
            flag_acq=flag_acq if (flag_acq >= 0 and pid == spec.patches[0].patch_id) else -1))
    return DecoderManifest(experiment=spec.name, program_fingerprint="", num_acq=len(lowered.acq_keys),
                           patches=patches, acq_to_patch_round_slot=acq_map)


def default_discriminator(spec: DeviceSpec, q: int) -> Discriminator:
    """Placeholder used only with allow_uncalibrated=True (will misclassify!)."""
    return Discriminator(c0=0j, c1=2 + 0j, c2=None, sigma=0.3)


class QecCompiler:
    def __init__(self, device: DeviceSpec, calibration: CalibrationStore, rng: Optional[RngFactory] = None):
        self.device = device
        self.cal = calibration
        self.rng = rng or RngFactory(12345)

    def control_settings(self) -> ControlSettings:
        cal = self.cal
        return ControlSettings(
            drive_freq_hz=np.array([q.drive_freq_hz for q in cal.qubits], float),
            sx_amp=np.array([q.sx_amp for q in cal.qubits], float),
            x180_amp=np.array([q.pi_amp for q in cal.qubits], float),
            cz_amp=np.array([c.cz_amp for c in cal.couplers], float),
            cz_corr=np.array([[c.corr0, c.corr1] for c in cal.couplers], float).reshape(-1, 2),
            version=cal.version)

    def discriminators(self, allow_uncalibrated: bool) -> list[Discriminator]:
        out = []
        for q, qc in enumerate(self.cal.qubits):
            if self.cal.readout_calibrated(q):
                out.append(Discriminator(c0=complex(*qc.c0), c1=complex(*qc.c1),
                                         c2=complex(*qc.c2) if qc.c2 is not None else None, sigma=qc.ro_sigma))
            elif allow_uncalibrated:
                out.append(default_discriminator(self.device, q))
            else:
                raise CompileError(f"qubit {q} ({self.device.qubits[q].name}) has no readout calibration; "
                                   "run calibration first or set allow_uncalibrated=True")
        return out

    def compile(self, exp: ExperimentSpec, snapshot: Optional[N.NoiseSnapshot] = None,
                meas_flip: Optional[list[float]] = None) -> CompiledExperiment:
        """Compile ``exp``.  ``snapshot`` overrides the calibration-derived noise
        model used for the DEM (e.g. a truth snapshot in validation tests)."""
        t0 = time.perf_counter()
        exp.validate()
        opts = exp.options
        discs = self.discriminators(opts.allow_uncalibrated)
        circ = build_logical_circuit(self.device, exp)
        lowered = lower(self.device, circ, opts, name=exp.name)
        prog = lowered.qpu
        ref = reference_sample(prog, self.rng.stream("compiler/reference/" + prog.fingerprint()))
        ref_acq = acq_bits(prog, ref)
        zann = z_eigen_annotations(prog, ref)
        flag_acq = -1
        if circ.flag_qubit is not None:
            for a, key in enumerate(lowered.acq_keys):
                if key[2] == "flag":
                    flag_acq = a
        manifest = build_detectors(lowered, exp, self.device, ref_acq, flag_acq)
        manifest.program_fingerprint = prog.fingerprint()
        snap = snapshot if snapshot is not None else estimated_snapshot(self.device, self.cal)
        t_dem = time.perf_counter()
        for pm in manifest.patches:
            dem = generate_dem(prog, pm.detectors, pm.observables, snap if opts.build_dem else None,
                               meas_flip=meas_flip, z_states=zann,
                               detector_names=[f"p{pm.patch_id}.D{i}" for i in range(pm.num_detectors)])
            pm.dem = dem
        t_end = time.perf_counter()
        stats = dict(lowered.stats)
        stats.update({"compile_s": round(t_end - t0, 4), "dem_s": round(t_end - t_dem, 4),
                      "schedule": circ.schedule_used,
                      "dem_mechanisms": {pm.patch_id: (pm.dem.num_mechanisms if pm.dem else None)
                                         for pm in manifest.patches},
                      "detectors": {pm.patch_id: pm.num_detectors for pm in manifest.patches}})
        return CompiledExperiment(spec=exp, device=self.device, qpu_program=prog, sequencers=lowered.sequencers,
                                  settings=self.control_settings(), discriminators=discs, manifest=manifest,
                                  lowered=lowered, noise_snapshot=snap, z_states=zann,
                                  calibration_version=self.cal.version,
                                  config_epoch=manifest.config_epoch(), stats=stats)

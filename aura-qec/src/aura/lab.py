"""
VirtualLab: the complete Layer 0 + Layer 1 system in one object.

  virtual QPU  <->  control system  <->  AURA-Link (PHY / gateway)  <->  decoder stand-in
       ^                  ^
       |                  |
  calibration graph    QEC compiler  <-  calibration store
       ^                                    ^
       +-------- operations console / CLI --+

Everything shares one deterministic virtual clock.  The lab is what the CLI,
the web console, the examples and the integration tests drive.
"""

from __future__ import annotations

import dataclasses
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .codes import get_code
from .codes.css import CSSCode
from .control.system import ControlSystem, ControlTiming
from .core.config import from_dict, load_file
from .core.errors import AuraError, ConfigError
from .core.rng import RngFactory
from .core.sim import Simulator
from .core.telemetry import MetricsRegistry
from .core.units import NS, US
from .host.calibration import CalibrationGraph, CalibrationStore, Calibrator, CalSettings
from .host.compiler import (CompileOptions, CompiledExperiment, ExperimentSpec, PatchSpec, QecCompiler)
from .host.runner.results import PatchResult, RunResult, latency_stats, wilson_interval
from .link import LinkConfig, build_sim_link
from .link.endpoint import LinkState
from .qpu import VirtualQPU, build_device
from .qpu.device import PROFILES
from .testbed import MatchingDecoderStub, NullDecoderStub, OracleDecoderStub


@dataclass
class PatchConfig:
    patch_id: int = 0
    code: str = "surface:3"


@dataclass
class LabConfig:
    seed: int = 1
    profile: str = "sota"
    patches: list[PatchConfig] = field(default_factory=lambda: [PatchConfig()])
    drift_speedup: float = 1.0
    cosmic_rays: bool = True
    feedline_size: int = 8
    decoder_latency_ns: float = 400.0
    decoder_per_event_ns: float = 20.0

    def validate(self) -> None:
        if self.profile not in PROFILES:
            raise ConfigError(f"unknown profile {self.profile}; choose from {sorted(PROFILES)}")
        if not self.patches:
            raise ConfigError("at least one patch is required")
        ids = [p.patch_id for p in self.patches]
        if len(ids) != len(set(ids)):
            raise ConfigError("duplicate patch ids")
        if self.drift_speedup <= 0:
            raise ConfigError("drift_speedup must be > 0")


DECODERS = {"null": NullDecoderStub, "oracle": OracleDecoderStub, "matching": MatchingDecoderStub}


class VirtualLab:
    def __init__(self, cfg: Optional[LabConfig] = None, link_cfg: Optional[LinkConfig] = None,
                 control_timing: Optional[ControlTiming] = None, cal_settings: Optional[CalSettings] = None):
        self.cfg = cfg or LabConfig()
        self.cfg.validate()
        self.lock = threading.RLock()
        self.rng = RngFactory(self.cfg.seed)
        self.sim = Simulator()
        self.codes: dict[int, CSSCode] = {p.patch_id: get_code(p.code) for p in self.cfg.patches}
        prof = PROFILES[self.cfg.profile]
        drift = dataclasses.replace(prof.drift, speedup=self.cfg.drift_speedup,
                                    cosmic_rate_per_s=prof.drift.cosmic_rate_per_s if self.cfg.cosmic_rays else 0.0)
        self.device = build_device(list(self.codes.items()), prof, seed=self.cfg.seed,
                                   feedline_size=self.cfg.feedline_size, drift=drift)
        self.spec = self.device.spec
        self.qpu = VirtualQPU(self.device, self.rng.child("qpu"), scheduler=self.sim)
        self.store = CalibrationStore(self.spec)
        self.calibrator = Calibrator(self.sim, self.qpu, self.store, self.rng.stream("calibration"),
                                     cal_settings, log=self._log)
        self.calgraph = CalibrationGraph(self.calibrator)
        self.compiler = QecCompiler(self.spec, self.store, self.rng.child("compiler"))
        self.metrics = MetricsRegistry("lab")
        self.control = ControlSystem(self.sim, self.qpu, control_timing or ControlTiming(),
                                     self.rng.child("control"), MetricsRegistry("control"))
        self.link_cfg = link_cfg or LinkConfig()
        self.link = build_sim_link(self.sim, self.link_cfg, self.rng.child("link"), sink=None,
                                   control_clock=self.control.clock)
        self.control.attach_link(self.link.control)
        self.link.control.on_correction = self.control.on_correction
        self.runs: list[RunResult] = []
        self.events: list[tuple[int, str]] = []
        self.last_compiled: Optional[CompiledExperiment] = None
        self.busy: Optional[str] = None
        self._log(f"lab up: {self.spec.name}, {self.spec.num_qubits} qubits, {len(self.spec.couplers)} couplers, "
                  f"profile={self.cfg.profile}")

    # ------------------------------------------------------------------ utils
    def _log(self, msg: str) -> None:
        self.events.append((self.sim.now, msg))
        if len(self.events) > 2000:
            del self.events[:500]

    @classmethod
    def from_file(cls, path: str) -> "VirtualLab":
        d = load_file(path)
        lab = from_dict(LabConfig, d.get("lab", {}))
        link = from_dict(LinkConfig, d["link"]) if "link" in d else None
        ctl = from_dict(ControlTiming, d["control"]) if "control" in d else None
        cal = from_dict(CalSettings, d["calibration"]) if "calibration" in d else None
        return cls(lab, link, ctl, cal)

    # ------------------------------------------------------------------ calibration
    def calibrate(self, mode: str = "full") -> dict:
        """mode: 'full' (bring-up), 'maintain' (stale / out-of-spec only),
        'oracle' (TEST ONLY: copy hidden truth into the store)."""
        with self.lock:
            self._require_idle()
            self.busy = f"calibrate:{mode}"
            try:
                t0 = time.perf_counter()
                if mode == "full":
                    rep = self.calgraph.calibrate_all()
                    summary = {"actions": rep.summary(), "virtual_ms": rep.virtual_time_ps / 1e9}
                elif mode == "maintain":
                    rep = self.calgraph.maintain()
                    summary = {"actions": rep.summary(), "virtual_ms": rep.virtual_time_ps / 1e9}
                elif mode == "oracle":
                    self._oracle_calibration()
                    summary = {"actions": "oracle (truth copied; test mode)", "virtual_ms": 0.0}
                else:
                    raise AuraError(f"unknown calibration mode {mode}")
                summary["wall_s"] = round(time.perf_counter() - t0, 2)
                summary["store"] = self.store.summary()
                self._log(f"calibration {mode}: {summary['actions']}")
                return summary
            finally:
                self.busy = None

    def _oracle_calibration(self) -> None:
        t = self.sim.now
        st = self.qpu.oracle_control_settings(t)
        snap = self.qpu.snapshot(t)
        for q, qt in enumerate(self.device.qubits):
            ro = qt.readout
            qn = snap.qubits[q]
            self.store.update_qubit(q, t, "oracle", drive_freq_hz=float(st.drive_freq_hz[q]),
                                    pi_amp=float(st.x180_amp[q]), sx_amp=float(st.sx_amp[q]),
                                    c0=(ro.c0.real, ro.c0.imag), c1=(ro.c1.real, ro.c1.imag),
                                    c2=(ro.c2.real, ro.c2.imag), ro_sigma=ro.sigma,
                                    t1_ps=qn.t1_ps, t2_star_ps=qn.t2_star_ps, t2_echo_ps=qn.t2_echo_ps,
                                    sq_error=qt.sq_depol)
        # readout assignment errors from the physical readout model (Monte Carlo)
        self.calibrator.readout()
        for c, ct in enumerate(self.device.couplers):
            self.store.update_coupler(c, t, "oracle", cz_amp=float(st.cz_amp[c]), corr0=float(st.cz_corr[c, 0]),
                                      corr1=float(st.cz_corr[c, 1]), cz_error=ct.cz_depol, cond_phase_rad=np.pi)

    # ------------------------------------------------------------------ compile
    def experiment(self, rounds: int = 3, basis: str = "Z", branching: bool = False,
                   patches: Optional[list[int]] = None, **opts) -> ExperimentSpec:
        pids = patches if patches is not None else sorted(self.codes)
        ps = tuple(PatchSpec(pid, self.codes[pid], basis, rounds) for pid in pids)
        name = "memory-" + "+".join(f"{self.codes[p].name}" for p in pids) + f"-{basis}-r{rounds}"
        return ExperimentSpec(name=name, patches=ps, options=CompileOptions(branching=branching, **opts))

    def compile(self, exp: ExperimentSpec) -> CompiledExperiment:
        with self.lock:
            ce = self.compiler.compile(exp)
            self.last_compiled = ce
            self._log(f"compiled {exp.name}: epoch {ce.config_epoch:#010x}, "
                      f"{ce.stats['ops']}, {ce.stats['shot_duration_ns']:.0f} ns/shot")
            return ce

    # ------------------------------------------------------------------ run
    def _require_idle(self) -> None:
        if self.busy:
            raise AuraError(f"lab is busy ({self.busy})")

    def run(self, exp: ExperimentSpec, shots: int = 100, decoder: str = "matching", mode: str = "stream",
            timeout_us: Optional[float] = None) -> RunResult:
        if decoder not in DECODERS:
            raise AuraError(f"unknown decoder {decoder}; choose from {sorted(DECODERS)}")
        if mode not in ("stream", "batch"):
            raise AuraError("mode must be 'stream' or 'batch'")
        if shots < 1:
            raise AuraError("shots must be >= 1")
        with self.lock:
            self._require_idle()
            self.busy = f"run:{exp.name}"
            try:
                ce = self.compile(exp)
                if mode == "stream":
                    return self._run_stream(ce, shots, decoder, timeout_us)
                return self._run_batch(ce, shots, decoder)
            finally:
                self.busy = None

    def _make_stub(self, decoder: str):
        cls = DECODERS[decoder]
        return cls(self.sim, None, base_latency_ps=int(self.cfg.decoder_latency_ns * NS),
                   per_event_ps=int(self.cfg.decoder_per_event_ns * NS))

    def _run_stream(self, ce: CompiledExperiment, shots: int, decoder: str, timeout_us: Optional[float]) -> RunResult:
        t_wall = time.perf_counter()
        stub = self._make_stub(decoder)
        stub.attach(self.link.gateway)
        self.link.gateway.sink = stub
        cs = self.control
        n_before = len(cs.records)
        for key in ("syn.meas_to_arrival_ps", "syn.meas_to_ready_ps", "syn.ingress_pipeline_ps"):
            self.link.gateway.m.histogram(key).reset()        # latency statistics are per run
        cs.load(ce)
        done = {"flag": False}
        start_t = self.sim.now

        def finished() -> None:
            done["flag"] = True
            # let in-flight corrections drain, then stop
            self.sim.schedule(200 * US, self.sim.stop)

        def ready() -> None:
            self.link.control.on_ready = None
            cs.run_shots(shots, on_done=finished)

        if self.link.control.state == LinkState.UP and self.link.control.epoch == ce.config_epoch \
                and self.link.gateway.manifest is not None:
            # session already configured for this exact program: hand the config to the new decoder
            stub.on_session("config", {"manifest": self.link.gateway.manifest, "epoch": ce.config_epoch})
            ready()
        else:
            self.link.control.on_ready = ready
            self.link.control.open(ce.manifest)
        per_shot = ce.qpu_program.shot_duration_ps + ce.spec.options.rep_delay_ps + 20 * US
        limit = int(timeout_us * US) if timeout_us else 5_000 * US + shots * per_shot * 4
        self.sim.run(until_ps=self.sim.now + limit)
        notes = []
        if not done["flag"]:
            notes.append(f"run did not complete within the virtual timeout ({limit / 1e6:.0f} us); "
                         f"link state {self.link.control.state.name}")
            self.link.control.on_ready = None
            if cs.running:
                # stop issuing shots and let the shot in flight finish cleanly (bounded by the
                # program length plus the feed-forward decision timeout)
                cs._shots_remaining = 0
                guard = ce.qpu_program.shot_duration_ps + ce.spec.options.decision_timeout_ps + 100 * US
                self.sim.run(until_ps=self.sim.now + guard)
                if cs.running and not self.qpu.shot_active:
                    cs.running = False
        recs = cs.records[n_before:]
        res = self._summarise(ce, recs, "stream", decoder, stub, notes)
        res.virtual_time_ps = self.sim.now - start_t
        res.wall_s = round(time.perf_counter() - t_wall, 3)
        res.link = {"control": self.link.control.stats(), "gateway": self.link.gateway.stats()}
        res.control = cs.metrics.snapshot()
        self.runs.append(res)
        self._log(res.summary().splitlines()[0])
        return res

    def _run_batch(self, ce: CompiledExperiment, shots: int, decoder: str) -> RunResult:
        t_wall = time.perf_counter()
        self.qpu.upload_settings(ce.settings)
        self.qpu.load_program(ce.qpu_program)
        out = self.qpu.run_batch(shots, t_start_ps=self.sim.now)
        bits = np.zeros((shots, ce.num_acq), dtype=np.uint8)
        for a, q in enumerate(ce.lowered.acq_qubit):
            b, _, _ = ce.discriminators[q].classify(out["iq"][:, a])
            bits[:, a] = b
        stub = self._make_stub(decoder)
        stub.on_session("config", {"manifest": ce.manifest, "epoch": ce.config_epoch})
        recs = []
        for s in range(shots):
            r = _BatchRecord(s, bits[s])
            for pm in ce.manifest.patches:
                obs, status, _ = stub.decode(pm, bits[s])
                r.corrections[pm.patch_id] = obs
            recs.append(r)
        # advance virtual time as the hardware would have
        dt = shots * (ce.qpu_program.shot_duration_ps + ce.spec.options.rep_delay_ps)
        self.sim.run(until_ps=self.sim.now + dt)
        res = self._summarise(ce, recs, "batch", decoder, stub, [])
        res.virtual_time_ps = dt
        res.wall_s = round(time.perf_counter() - t_wall, 3)
        self.runs.append(res)
        self._log(res.summary().splitlines()[0])
        return res

    def _summarise(self, ce: CompiledExperiment, recs, mode: str, decoder: str, stub, notes) -> RunResult:
        run_id = uuid.uuid4().hex[:10]
        bits = np.array([r.bits for r in recs], dtype=np.uint8) if recs else np.zeros((0, ce.num_acq), np.uint8)
        patches = []
        lat_meas_to_cor = []
        for pm in ce.manifest.patches:
            n = len(recs)
            if n == 0:
                continue
            raw = pm.observable_flips(bits)
            corr = np.zeros_like(raw)
            missing = 0
            for i, r in enumerate(recs):
                c = r.corrections.get(pm.patch_id)
                if c is None:
                    missing += 1
                    continue
                corr[i] = c.obs_flips if hasattr(c, "obs_flips") else c
                if hasattr(r, "t_correction") and pm.patch_id in r.t_correction and r.t_last_meas_end >= 0:
                    lat_meas_to_cor.append(r.t_correction[pm.patch_id] - r.t_last_meas_end)
            final = raw ^ corr
            fail = final.any(axis=1)
            k = int(fail.sum())
            ev = pm.detection_events(bits)
            pred = float(pm.dem.detector_marginals().mean()) if pm.dem is not None else None
            ler = k / n
            per_round = None
            if 0 < ler < 0.5:
                per_round = float(0.5 * (1 - (1 - 2 * ler) ** (1.0 / pm.rounds)))
            patches.append(PatchResult(
                patch_id=pm.patch_id, code=pm.code_name, basis=pm.basis, rounds=pm.rounds, shots=n,
                raw_logical_errors=int(raw.any(axis=1).sum()), logical_errors=k, missing_corrections=missing,
                detection_rate=float(ev.mean()) if ev.size else 0.0, predicted_detection_rate=pred,
                per_observable_errors=[int(x) for x in final.sum(axis=0)], ler=ler,
                ler_ci=wilson_interval(k, n), ler_per_round=per_round))
            if missing:
                notes.append(f"patch {pm.patch_id}: {missing} shots without a correction "
                             "(counted as the uncorrected outcome)")
        latency = {}
        if lat_meas_to_cor:
            latency["measurement_to_correction_ps"] = latency_stats(lat_meas_to_cor)
        if mode == "stream":
            gw = self.link.gateway.m
            for key in ("syn.meas_to_arrival_ps", "syn.meas_to_ready_ps", "syn.ingress_pipeline_ps"):
                h = gw.histogram(key)
                if h.count:
                    latency[key.replace("syn.", "")] = h.snapshot()
        branching = {}
        if mode == "stream" and any(pm.flag_acq >= 0 for pm in ce.manifest.patches):
            dec = [r.decisions for r in recs if r.decisions]
            if dec:
                resp = []
                timeouts = 0
                for r in recs:
                    for reg, (val, t_applied, to) in r.decisions.items():
                        timeouts += int(bool(to))
                        if t_applied is not None and r.t_last_meas_end >= 0:
                            resp.append(t_applied - r.t_last_meas_end)
                flag_ok = None
                pm0 = ce.manifest.patches[0]
                if pm0.flag_acq >= 0:
                    agree = [int(r.bits[pm0.flag_acq]) == int(r.decisions.get(f"L{pm0.patch_id}", [0])[0])
                             for r in recs if r.decisions]
                    flag_ok = float(np.mean(agree)) if agree else None
                branching = {"response_ps": latency_stats(resp), "timeouts": timeouts,
                             "flag_matches_decision": flag_ok}
        res = RunResult(run_id=run_id, experiment=ce.spec.name, mode=mode, decoder=decoder, shots=len(recs),
                        patches=patches, latency=latency, decoder_stats=stub.m.snapshot(), branching=branching,
                        calibration_version=ce.calibration_version, config_epoch=ce.config_epoch,
                        compile_stats=ce.stats, notes=notes)
        return res

    # ------------------------------------------------------------------ status (console)
    def status(self) -> dict:
        """Console status.  Never blocks: while a job holds the lab, the last
        complete status is returned with the current ``busy`` flag."""
        if not self.lock.acquire(blocking=False):
            st = dict(getattr(self, "_last_status", {}) or {})
            st["busy"] = self.busy
            st["stale"] = True
            return st
        try:
            st = self._status()
            self._last_status = st
            return st
        finally:
            self.lock.release()

    def _status(self) -> dict:
        snap = self.qpu.snapshot(self.sim.now)
        qubits = []
        for q, info in enumerate(self.spec.qubits):
            qc = self.store.qubits[q]
            qubits.append({
                "index": q, "name": info.name, "role": info.role, "coords": info.coords,
                "t1_us_est": (qc.t1_ps or 0) / 1e6 or None,
                "t1_us_true": snap.qubits[q].t1_ps / 1e6,
                "readout_err_est": ((qc.p01 + qc.p10) / 2) if qc.p01 is not None and qc.p10 is not None else None,
                "detuning_khz_true": snap.qubits[q].detuning_hz / 1e3,
                "calibrated": self.store.readout_calibrated(q),
            })
        return {
            "virtual_time_us": self.sim.now / 1e6,
            "busy": self.busy,
            "device": {"name": self.spec.name, "qubits": self.spec.num_qubits, "couplers": len(self.spec.couplers),
                       "profile": self.cfg.profile, "patches": {p: c.label + " " + c.name for p, c in self.codes.items()}},
            "calibration": self.store.summary(),
            "link": {"control": self.link.control.state.name, "gateway": self.link.gateway.state.name,
                     "epoch": self.link.control.epoch,
                     "time_offset_ps": self.link.gateway.ts_slave.last_offset_ps,
                     "path_delay_ps": self.link.gateway.ts_slave.last_delay_ps,
                     "credits": self.link.control.credits_available},
            "cosmic_events": len(self.qpu.drift.cosmic_history),
            "qubits": qubits,
            "runs": [{"run_id": r.run_id, "experiment": r.experiment, "mode": r.mode, "decoder": r.decoder,
                      "shots": r.shots, "ler": [p.ler for p in r.patches]} for r in self.runs[-20:]],
            "events": [{"t_us": t / 1e6, "msg": m} for t, m in self.events[-50:]],
        }


class _BatchRecord:
    __slots__ = ("shot", "bits", "corrections", "decisions", "t_last_meas_end")

    def __init__(self, shot: int, bits: np.ndarray):
        self.shot = shot
        self.bits = bits
        self.corrections: dict = {}
        self.decisions: dict = {}
        self.t_last_meas_end = -1

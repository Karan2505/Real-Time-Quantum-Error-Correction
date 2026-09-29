"""
Physical device model of the virtual superconducting QPU.

``DeviceTruth`` holds the *hidden* physical parameters of every transmon
qubit and tunable coupler (the things a lab never knows exactly).  The host
only ever sees ``DeviceSpec`` -- topology, design values and nominal
durations -- and must *measure* everything else through calibration.

Physical model (per qubit): flux-tunable transmon with f01, anharmonicity,
T1, T2*, T2-echo, drive amplitude needed for a perfect pi pulse, residual
gate error, leakage, dispersive readout (IQ blob per state |0>,|1>,|2>),
reset infidelity.  Per coupler: CZ gate via flux pulse (the conditional
phase depends on the flux amplitude), CZ error, CZ leakage, spectator error.

Parameter profiles are *representative* of published 2024-2026
superconducting hardware (T1 ~ 50-100 us, 1q error ~ 1e-4..1e-3, CZ error
~ 2e-3..1e-2, readout error ~ 0.5-2 %).  They are not any vendor's data.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..codes.css import CSSCode
from ..core.errors import ConfigError
from ..core.units import NS, US


# =========================================================================== topology
@dataclass(frozen=True)
class QubitInfo:
    index: int
    name: str
    role: str            # 'data' | 'x_ancilla' | 'z_ancilla' | 'spare'
    patch: int
    local_index: int     # data index within the code, or check index
    coords: tuple[float, float]
    feedline: int        # multiplexed readout line


@dataclass(frozen=True)
class CouplerInfo:
    index: int
    name: str
    q0: int
    q1: int


@dataclass(frozen=True)
class Durations:
    sx_ps: int = 24 * NS
    x180_ps: int = 24 * NS
    cz_ps: int = 48 * NS
    meas_ps: int = 500 * NS
    reset_ps: int = 160 * NS
    grid_ps: int = 4 * NS          # sequencer timing grid (250 MHz)

    def validate(self) -> None:
        for f in dataclasses.fields(self):
            v = getattr(self, f.name)
            if v <= 0:
                raise ConfigError(f"duration {f.name} must be positive")
            if f.name != "grid_ps" and v % self.grid_ps:
                raise ConfigError(f"duration {f.name}={v} ps is not a multiple of the {self.grid_ps} ps grid")


@dataclass(frozen=True)
class DeviceSpec:
    """Public view of the device (what the host / compiler may know)."""
    name: str
    qubits: tuple[QubitInfo, ...]
    couplers: tuple[CouplerInfo, ...]
    durations: Durations
    design_f01_hz: tuple[float, ...]       # fab design targets (true values differ)
    design_pi_amp: tuple[float, ...]
    design_cz_amp: tuple[float, ...]
    feedline_size: int = 8

    @property
    def num_qubits(self) -> int:
        return len(self.qubits)

    def coupler_index(self) -> dict[tuple[int, int], int]:
        out = {}
        for c in self.couplers:
            out[(c.q0, c.q1)] = c.index
            out[(c.q1, c.q0)] = c.index
        return out

    def neighbors(self) -> list[list[int]]:
        nb = [[] for _ in self.qubits]
        for c in self.couplers:
            nb[c.q0].append(c.q1)
            nb[c.q1].append(c.q0)
        return nb

    def qubit_by_name(self) -> dict[str, int]:
        return {q.name: q.index for q in self.qubits}

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "num_qubits": self.num_qubits,
            "qubits": [dataclasses.asdict(q) for q in self.qubits],
            "couplers": [dataclasses.asdict(c) for c in self.couplers],
            "durations": dataclasses.asdict(self.durations),
            "feedline_size": self.feedline_size,
        }


def code_native_topology(patches: list[tuple[int, CSSCode]], feedline_size: int = 8,
                         patch_spacing: float = 4.0) -> tuple[list[QubitInfo], list[CouplerInfo]]:
    """Qubits (data + one ancilla per check) and one tunable coupler per
    (ancilla, data) pair needed by the syndrome-extraction circuit.

    This is a *code-native* chip: the connectivity is exactly the Tanner graph
    of the code (as in planar surface-code processors, or the two-layer
    long-range-coupler designs proposed for bivariate-bicycle codes).
    """
    qubits: list[QubitInfo] = []
    couplers: list[CouplerInfo] = []
    x_off = 0.0
    seen_ids = set()
    for pid, code in patches:
        if pid in seen_ids:
            raise ConfigError(f"duplicate patch id {pid}")
        seen_ids.add(pid)
        coords = code.data_coords if code.data_coords is not None else np.array(
            [[2 * i, 0] for i in range(code.n)], float)
        xcc = code.x_check_coords if code.x_check_coords is not None and len(code.x_check_coords) else \
            np.array([[2 * i, 1] for i in range(code.num_x_checks)], float).reshape(-1, 2)
        zcc = code.z_check_coords if code.z_check_coords is not None and len(code.z_check_coords) else \
            np.array([[2 * i, -1] for i in range(code.num_z_checks)], float).reshape(-1, 2)
        width = float(max([c[0] for c in coords] + [0])) + patch_spacing
        base = len(qubits)
        for i in range(code.n):
            qubits.append(QubitInfo(len(qubits), f"p{pid}.d{i}", "data", pid, i,
                                    (float(coords[i][0]) + x_off, float(coords[i][1])), 0))
        xbase = len(qubits)
        for c in range(code.num_x_checks):
            qubits.append(QubitInfo(len(qubits), f"p{pid}.x{c}", "x_ancilla", pid, c,
                                    (float(xcc[c][0]) + x_off, float(xcc[c][1])), 0))
        zbase = len(qubits)
        for c in range(code.num_z_checks):
            qubits.append(QubitInfo(len(qubits), f"p{pid}.z{c}", "z_ancilla", pid, c,
                                    (float(zcc[c][0]) + x_off, float(zcc[c][1])), 0))
        for c in range(code.num_x_checks):
            for q in np.nonzero(code.hx[c])[0]:
                couplers.append(CouplerInfo(len(couplers), f"p{pid}.cx{c}_d{q}", xbase + c, base + int(q)))
        for c in range(code.num_z_checks):
            for q in np.nonzero(code.hz[c])[0]:
                couplers.append(CouplerInfo(len(couplers), f"p{pid}.cz{c}_d{q}", zbase + c, base + int(q)))
        x_off += width
    # multiplexed readout: consecutive qubits share a feedline
    qubits = [dataclasses.replace(q, feedline=q.index // feedline_size) for q in qubits]
    return qubits, couplers


# =========================================================================== truth params
@dataclass
class ReadoutTruth:
    c0: complex
    c1: complex
    c2: complex
    sigma: float                 # rms noise per quadrature of the integrated signal
    excite_prob: float           # measurement-induced |0> -> |1> transition
    rotation_drift_rad_per_s: float = 0.0


@dataclass
class QubitTruth:
    f01_hz: float
    anharm_hz: float
    t1_ps: float
    t2_star_ps: float
    t2_echo_ps: float
    pi_amp: float                # drive amplitude for a perfect pi rotation
    ef_pi_amp: float             # amplitude for a pi rotation on the |1>-|2> transition
    sq_depol: float              # residual 1q depolarising error beyond decoherence
    leak_sq: float               # leakage per single-qubit pulse
    reset_error: float           # P(|1>) after unconditional reset
    reset_leak_removal: float    # P(leakage removed by a reset)
    readout: ReadoutTruth


@dataclass
class CouplerTruth:
    cz_amp: float                # flux amplitude giving exactly pi conditional phase
    cz_depol: float
    leak_cz: float               # per participating qubit
    spectator_z: float           # Z error on each neighbouring (spectator) qubit
    dyn_phase0: float = 0.0      # single-qubit phase picked up by q0 during the CZ flux pulse
    dyn_phase1: float = 0.0      # ... by q1 (the controller must cancel these with virtual Z)


@dataclass
class DriftParams:
    """Temporal fluctuations (see qpu/drift.py)."""
    t1_log_sigma: float = 0.15            # stationary std of log(T1)
    t1_tau_s: float = 600.0               # OU correlation time
    freq_sigma_hz: float = 15e3           # stationary std of f01 fluctuation
    freq_tau_s: float = 300.0
    amp_sigma: float = 0.004              # relative pi-amplitude drift
    amp_tau_s: float = 1800.0
    cz_amp_sigma: float = 0.003
    cz_amp_tau_s: float = 1800.0
    readout_rot_sigma_rad: float = 0.02
    readout_tau_s: float = 900.0
    tls_rate_per_qubit_per_s: float = 1e-4   # sudden T1 dips from two-level-system defects
    tls_t1_factor: float = 0.3
    tls_duration_s: float = 60.0
    cosmic_rate_per_s: float = 0.1           # correlated error bursts (McEwen et al. 2022)
    cosmic_radius: int = 3                    # coupler-graph hops affected
    cosmic_t1_suppression: float = 50.0       # T1 -> T1 / (1 + A) at impact
    cosmic_tau_s: float = 0.025
    speedup: float = 1.0                      # accelerate drift for demos (1 = realistic)
    grid_s: float = 0.001                     # time resolution of drift processes

    def validate(self) -> None:
        if self.speedup <= 0 or self.grid_s <= 0:
            raise ConfigError("drift speedup and grid must be positive")


@dataclass
class NoiseProfile:
    name: str
    t1_us_median: float
    t1_log_sigma: float
    t2_star_ratio: tuple[float, float]      # T2*/T1 uniform range (capped at 2 T1)
    t2_echo_ratio: tuple[float, float]      # T2echo/T1
    sq_depol_median: float
    cz_depol_median: float
    cz_depol_log_sigma: float
    leak_cz: float
    leak_sq: float
    spectator_z: float
    readout_sigma: float                    # blob sigma (blob separation is 2.0)
    readout_excite: float
    reset_error: float
    reset_leak_removal: float
    f01_spread_hz: float = 50e6             # true f01 vs design target
    amp_spread: float = 0.05                # true pi amp vs design
    cz_amp_spread: float = 0.05
    drift: DriftParams = field(default_factory=DriftParams)


PROFILES: dict[str, NoiseProfile] = {
    # Aggressive but published-order-of-magnitude 2025-26 superconducting device.
    "sota": NoiseProfile("sota", t1_us_median=85.0, t1_log_sigma=0.2, t2_star_ratio=(0.6, 1.2),
                         t2_echo_ratio=(1.2, 1.8), sq_depol_median=2.5e-4, cz_depol_median=2.5e-3,
                         cz_depol_log_sigma=0.3, leak_cz=2e-4, leak_sq=1e-6, spectator_z=1e-4,
                         readout_sigma=0.30, readout_excite=5e-4, reset_error=1e-3,
                         reset_leak_removal=0.95),
    # Mid-range device (roughly early-2020s cloud hardware).
    "nisq": NoiseProfile("nisq", t1_us_median=30.0, t1_log_sigma=0.3, t2_star_ratio=(0.4, 1.0),
                         t2_echo_ratio=(0.8, 1.5), sq_depol_median=1e-3, cz_depol_median=1e-2,
                         cz_depol_log_sigma=0.4, leak_cz=1e-3, leak_sq=1e-5, spectator_z=5e-4,
                         readout_sigma=0.42, readout_excite=2e-3, reset_error=5e-3,
                         reset_leak_removal=0.9),
    # Noise-free device (for plumbing tests); coherence effectively infinite.
    "ideal": NoiseProfile("ideal", t1_us_median=1e12, t1_log_sigma=0.0, t2_star_ratio=(2.0, 2.0),
                          t2_echo_ratio=(2.0, 2.0), sq_depol_median=0.0, cz_depol_median=0.0,
                          cz_depol_log_sigma=0.0, leak_cz=0.0, leak_sq=0.0, spectator_z=0.0,
                          readout_sigma=1e-6, readout_excite=0.0, reset_error=0.0,
                          reset_leak_removal=1.0, f01_spread_hz=0.0, amp_spread=0.0, cz_amp_spread=0.0,
                          drift=DriftParams(t1_log_sigma=0.0, freq_sigma_hz=0.0, amp_sigma=0.0,
                                            cz_amp_sigma=0.0, readout_rot_sigma_rad=0.0,
                                            tls_rate_per_qubit_per_s=0.0, cosmic_rate_per_s=0.0)),
}


@dataclass
class DeviceTruth:
    spec: DeviceSpec
    qubits: list[QubitTruth]
    couplers: list[CouplerTruth]
    drift: DriftParams
    profile: str
    seed: int

    @property
    def n(self) -> int:
        return len(self.qubits)


def build_device(patches: list[tuple[int, CSSCode]], profile: str | NoiseProfile = "sota",
                 seed: int = 1, durations: Optional[Durations] = None, feedline_size: int = 8,
                 name: Optional[str] = None, drift: Optional[DriftParams] = None) -> DeviceTruth:
    """Fabricate a virtual chip for the given code patches."""
    prof = PROFILES[profile] if isinstance(profile, str) else profile
    if isinstance(profile, str) and profile not in PROFILES:
        raise ConfigError(f"unknown profile {profile!r}; choose from {sorted(PROFILES)}")
    dur = durations or Durations()
    dur.validate()
    qinfo, cinfo = code_native_topology(patches, feedline_size=feedline_size)
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0xD3F1CE]))
    nq = len(qinfo)
    design_f = [4.8e9 + 0.12e9 * ((i * 7) % 11) / 10.0 for i in range(nq)]
    design_amp = [0.5] * nq
    design_cz = [0.8] * len(cinfo)
    qt: list[QubitTruth] = []
    for i in range(nq):
        t1 = prof.t1_us_median * math.exp(rng.normal(0, prof.t1_log_sigma)) * US
        t2s = min(2 * t1, t1 * rng.uniform(*prof.t2_star_ratio))
        t2e = min(2 * t1, max(t2s, t1 * rng.uniform(*prof.t2_echo_ratio)))
        # IQ blobs: |0> at 0, |1> at distance 2 (random orientation), |2> beyond |1>
        ang = rng.uniform(0, 2 * math.pi)
        c0 = complex(rng.normal(0, 0.05), rng.normal(0, 0.05))
        c1 = c0 + 2.0 * complex(math.cos(ang), math.sin(ang))
        c2 = c0 + 2.0 * complex(math.cos(ang + 0.5), math.sin(ang + 0.5)) * 1.6
        sq = prof.sq_depol_median * math.exp(rng.normal(0, 0.3)) if prof.sq_depol_median else 0.0
        qt.append(QubitTruth(
            f01_hz=design_f[i] + rng.normal(0, prof.f01_spread_hz) * 0.02,
            anharm_hz=-210e6 + rng.normal(0, 5e6),
            t1_ps=t1, t2_star_ps=t2s, t2_echo_ps=t2e,
            pi_amp=(pa := design_amp[i] * (1 + rng.normal(0, prof.amp_spread))),
            ef_pi_amp=pa / math.sqrt(2) * (1 + rng.normal(0, 0.01)),
            sq_depol=sq, leak_sq=prof.leak_sq,
            reset_error=prof.reset_error, reset_leak_removal=prof.reset_leak_removal,
            readout=ReadoutTruth(c0=c0, c1=c1, c2=c2,
                                 sigma=prof.readout_sigma * math.exp(rng.normal(0, 0.08)),
                                 excite_prob=prof.readout_excite)))
    ct: list[CouplerTruth] = []
    for c in cinfo:
        cz = prof.cz_depol_median * math.exp(rng.normal(0, prof.cz_depol_log_sigma)) \
            if prof.cz_depol_median else 0.0
        ideal = prof.name == "ideal"
        ct.append(CouplerTruth(cz_amp=design_cz[c.index] * (1 + rng.normal(0, prof.cz_amp_spread)),
                               cz_depol=min(cz, 0.2), leak_cz=prof.leak_cz,
                               spectator_z=prof.spectator_z,
                               dyn_phase0=0.0 if ideal else float(rng.uniform(-math.pi, math.pi)),
                               dyn_phase1=0.0 if ideal else float(rng.uniform(-math.pi, math.pi))))
    spec = DeviceSpec(name=name or "aura-vqpu-" + "-".join(f"p{p}" for p, _ in patches),
                      qubits=tuple(qinfo), couplers=tuple(cinfo), durations=dur,
                      design_f01_hz=tuple(design_f), design_pi_amp=tuple(design_amp),
                      design_cz_amp=tuple(design_cz), feedline_size=feedline_size)
    dp = drift or prof.drift
    dp.validate()
    return DeviceTruth(spec=spec, qubits=qt, couplers=ct, drift=dp, profile=prof.name, seed=seed)

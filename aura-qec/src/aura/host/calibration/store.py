"""
Calibration store: the host's *belief* about the device.

Everything the compiler needs (pulse amplitudes, drive frequencies, CZ flux
amplitudes and phase compensations, readout discriminators, coherence and
error estimates) lives here with a timestamp and provenance.  Versioned
snapshots are JSON-serialisable so the operations console can show history
and runs can record exactly which calibration they used.
"""

from __future__ import annotations

import copy
import json
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from ...qpu.device import DeviceSpec


@dataclass
class QubitCal:
    drive_freq_hz: float
    pi_amp: float
    sx_amp: float
    t1_ps: Optional[float] = None
    t2_star_ps: Optional[float] = None
    t2_echo_ps: Optional[float] = None
    sq_error: Optional[float] = None            # estimated depolarising error per pulse
    # readout discriminator (Gaussian blobs, shared isotropic sigma)
    c0: Optional[tuple[float, float]] = None
    c1: Optional[tuple[float, float]] = None
    c2: Optional[tuple[float, float]] = None
    ro_sigma: Optional[float] = None
    p01: Optional[float] = None                 # P(read 1 | prep 0)
    p10: Optional[float] = None                 # P(read 0 | prep 1)
    updated_ps: dict = field(default_factory=dict)   # parameter group -> virtual time of last update


@dataclass
class CouplerCal:
    cz_amp: float
    corr0: float = 0.0                           # virtual-Z compensation on q0 after CZ (rad)
    corr1: float = 0.0
    cz_error: Optional[float] = None
    cond_phase_rad: Optional[float] = None       # last measured conditional phase
    updated_ps: dict = field(default_factory=dict)


@dataclass
class CalibrationSnapshot:
    version: int
    t_ps: int
    qubits: list[QubitCal]
    couplers: list[CouplerCal]
    notes: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)

    @classmethod
    def from_json(cls, s: str) -> "CalibrationSnapshot":
        d = json.loads(s)

        def tup(v):
            return tuple(v) if isinstance(v, list) else v
        qs = []
        for q in d["qubits"]:
            for k in ("c0", "c1", "c2"):
                q[k] = tup(q.get(k))
            qs.append(QubitCal(**q))
        cs = [CouplerCal(**c) for c in d["couplers"]]
        return cls(version=d["version"], t_ps=d["t_ps"], qubits=qs, couplers=cs, notes=d.get("notes", ""))


class CalibrationStore:
    """Thread-safe, versioned calibration database."""

    def __init__(self, spec: DeviceSpec):
        self.spec = spec
        self._lock = threading.RLock()
        self.qubits = [QubitCal(drive_freq_hz=f, pi_amp=a, sx_amp=a / 2.0)
                       for f, a in zip(spec.design_f01_hz, spec.design_pi_amp)]
        self.couplers = [CouplerCal(cz_amp=a) for a in spec.design_cz_amp]
        self.version = 0
        self.history: list[tuple[int, int, str]] = []   # (version, t_ps, what)

    # ------------------------------------------------------------ updates
    def update_qubit(self, q: int, t_ps: int, group: str, **values) -> None:
        with self._lock:
            qc = self.qubits[q]
            for k, v in values.items():
                if not hasattr(qc, k):
                    raise KeyError(f"QubitCal has no field {k}")
                setattr(qc, k, v)
            qc.updated_ps[group] = t_ps
            self.version += 1
            self.history.append((self.version, t_ps, f"q{q}:{group}"))

    def update_coupler(self, c: int, t_ps: int, group: str, **values) -> None:
        with self._lock:
            cc = self.couplers[c]
            for k, v in values.items():
                if not hasattr(cc, k):
                    raise KeyError(f"CouplerCal has no field {k}")
                setattr(cc, k, v)
            cc.updated_ps[group] = t_ps
            self.version += 1
            self.history.append((self.version, t_ps, f"c{c}:{group}"))

    # ------------------------------------------------------------ queries
    def snapshot(self, t_ps: int = 0, notes: str = "") -> CalibrationSnapshot:
        with self._lock:
            return CalibrationSnapshot(self.version, t_ps, copy.deepcopy(self.qubits),
                                       copy.deepcopy(self.couplers), notes)

    def restore(self, snap: CalibrationSnapshot) -> None:
        with self._lock:
            if len(snap.qubits) != len(self.qubits) or len(snap.couplers) != len(self.couplers):
                raise ValueError("snapshot is for a different device")
            self.qubits = copy.deepcopy(snap.qubits)
            self.couplers = copy.deepcopy(snap.couplers)
            self.version += 1
            self.history.append((self.version, snap.t_ps, "restore"))

    def age_ps(self, now_ps: int, q: Optional[int] = None, group: str = "", coupler: Optional[int] = None) -> float:
        with self._lock:
            src = self.qubits[q] if q is not None else self.couplers[coupler]
            t = src.updated_ps.get(group)
            return float("inf") if t is None else float(now_ps - t)

    def readout_calibrated(self, q: int) -> bool:
        qc = self.qubits[q]
        return qc.c0 is not None and qc.c1 is not None and qc.ro_sigma is not None

    def fully_calibrated(self) -> bool:
        return all(self.readout_calibrated(q) for q in range(len(self.qubits)))

    def save(self, path: str | Path, t_ps: int = 0) -> None:
        Path(path).write_text(self.snapshot(t_ps).to_json())

    def load(self, path: str | Path) -> None:
        self.restore(CalibrationSnapshot.from_json(Path(path).read_text()))

    def summary(self) -> dict:
        with self._lock:
            def med(vals):
                v = [x for x in vals if x is not None]
                return float(np.median(v)) if v else None
            return {
                "version": self.version,
                "qubits": len(self.qubits),
                "couplers": len(self.couplers),
                "readout_calibrated": sum(self.readout_calibrated(q) for q in range(len(self.qubits))),
                "median_t1_us": (med([q.t1_ps for q in self.qubits]) or 0) / 1e6 or None,
                "median_readout_error": med([(q.p01 + q.p10) / 2 if q.p01 is not None and q.p10 is not None
                                             else None for q in self.qubits]),
                "median_cz_error": med([c.cz_error for c in self.couplers]),
            }

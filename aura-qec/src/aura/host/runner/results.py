"""Run results and statistics."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def _pct(a: np.ndarray, q: float) -> Optional[float]:
    return float(np.percentile(a, q)) if len(a) else None


@dataclass
class PatchResult:
    patch_id: int
    code: str
    basis: str
    rounds: int
    shots: int
    raw_logical_errors: int
    logical_errors: int                     # after decoder correction (any observable flipped)
    missing_corrections: int
    detection_rate: float
    predicted_detection_rate: Optional[float]
    per_observable_errors: list[int]
    ler: float
    ler_ci: tuple[float, float]
    ler_per_round: Optional[float]

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d["ler_ci"] = list(self.ler_ci)
        return d


@dataclass
class RunResult:
    run_id: str
    experiment: str
    mode: str
    decoder: str
    shots: int
    patches: list[PatchResult]
    latency: dict = field(default_factory=dict)
    link: dict = field(default_factory=dict)
    control: dict = field(default_factory=dict)
    decoder_stats: dict = field(default_factory=dict)
    branching: dict = field(default_factory=dict)
    calibration_version: int = 0
    config_epoch: int = 0
    virtual_time_ps: int = 0
    wall_s: float = 0.0
    compile_stats: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d["patches"] = [p.to_dict() for p in self.patches]
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True, default=_json_default)

    def summary(self) -> str:
        lines = [f"run {self.run_id}: {self.experiment} [{self.mode}, decoder={self.decoder}] "
                 f"{self.shots} shots, {self.virtual_time_ps / 1e6:.1f} us virtual, {self.wall_s:.1f} s wall"]
        for p in self.patches:
            lo, hi = p.ler_ci
            lines.append(f"  patch {p.patch_id} {p.code} {p.basis}-basis r={p.rounds}: LER {p.ler:.4f} "
                         f"[{lo:.4f}, {hi:.4f}]  raw {p.raw_logical_errors}/{p.shots}  "
                         f"det-rate {p.detection_rate:.4f} (model {p.predicted_detection_rate or float('nan'):.4f})"
                         f"  missing-cor {p.missing_corrections}")
        for k, v in self.latency.items():
            if isinstance(v, dict) and v.get("p50") is not None:
                lines.append(f"  latency {k}: p50 {v['p50'] / 1e3:.0f} ns  p99 {v['p99'] / 1e3:.0f} ns  "
                             f"max {v['max'] / 1e3:.0f} ns")
        for n in self.notes:
            lines.append(f"  note: {n}")
        return "\n".join(lines)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, complex):
        return [o.real, o.imag]
    return str(o)


def latency_stats(values_ps: list[int]) -> dict:
    a = np.asarray(values_ps, dtype=float)
    if not len(a):
        return {"count": 0, "p50": None, "p90": None, "p99": None, "max": None, "mean": None}
    return {"count": int(len(a)), "mean": float(a.mean()), "p50": _pct(a, 50), "p90": _pct(a, 90),
            "p99": _pct(a, 99), "max": float(a.max()), "min": float(a.min())}

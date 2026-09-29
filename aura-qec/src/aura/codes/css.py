"""
CSS stabilizer codes.

A CSS code is defined by parity-check matrices ``hx`` (X-type stabilizers,
detect Z errors) and ``hz`` (Z-type stabilizers, detect X errors) with
``hx @ hz.T == 0 (mod 2)``.  Logical operators are computed (or supplied)
and put in symplectic canonical form  lx @ lz.T == I.
"""

from __future__ import annotations

import dataclasses
import hashlib
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..core.errors import CodeError
from . import gf2


@dataclass(frozen=True)
class CheckSchedule:
    """Interleaved CNOT schedule for simultaneous X/Z check measurement.

    ``x_steps[c][t]`` / ``z_steps[c][t]``: data qubit touched by X/Z check ``c``
    at CNOT time step ``t`` (-1 = idle).  All checks share the same number of
    steps.  The schedule must measure the intended stabilizers (validated by the
    compiler's noiseless determinism check).
    """
    x_steps: tuple[tuple[int, ...], ...]
    z_steps: tuple[tuple[int, ...], ...]

    @property
    def depth(self) -> int:
        rows = self.x_steps + self.z_steps
        return len(rows[0]) if rows else 0


@dataclass(frozen=True, eq=False)
class CSSCode:
    name: str
    hx: np.ndarray
    hz: np.ndarray
    logical_x: np.ndarray
    logical_z: np.ndarray
    distance: Optional[int] = None
    data_coords: Optional[np.ndarray] = None       # (n, 2) float
    x_check_coords: Optional[np.ndarray] = None    # (mx, 2)
    z_check_coords: Optional[np.ndarray] = None    # (mz, 2)
    schedule: Optional[CheckSchedule] = None
    family: str = "generic"
    params: dict = field(default_factory=dict)

    # ----------------------------------------------------------------- props
    @property
    def n(self) -> int:
        return int(self.hx.shape[1] if self.hx.size else self.hz.shape[1])

    @property
    def k(self) -> int:
        return int(self.logical_x.shape[0])

    @property
    def num_x_checks(self) -> int:
        return int(self.hx.shape[0])

    @property
    def num_z_checks(self) -> int:
        return int(self.hz.shape[0])

    @property
    def label(self) -> str:
        d = self.distance if self.distance is not None else "?"
        return f"[[{self.n},{self.k},{d}]]"

    def fingerprint(self) -> str:
        """Stable content hash (used in config epochs)."""
        h = hashlib.blake2b(digest_size=16)
        for a in (self.hx, self.hz, self.logical_x, self.logical_z):
            h.update(np.ascontiguousarray(a, dtype=np.uint8).tobytes())
            h.update(str(a.shape).encode())
        return h.hexdigest()

    # ------------------------------------------------------------ validation
    def validate(self) -> "CSSCode":
        hx, hz, lx, lz = self.hx, self.hz, self.logical_x, self.logical_z
        n = self.n
        for nm, a in (("hx", hx), ("hz", hz), ("logical_x", lx), ("logical_z", lz)):
            if a.dtype != np.uint8 or a.ndim != 2:
                raise CodeError(f"{self.name}: {nm} must be a 2-D uint8 array")
            if a.size and a.shape[1] != n:
                raise CodeError(f"{self.name}: {nm} has {a.shape[1]} columns, expected {n}")
            if a.size and a.max() > 1:
                raise CodeError(f"{self.name}: {nm} is not binary")
        if hx.size and hz.size and gf2.matmul(hx, hz.T).any():
            raise CodeError(f"{self.name}: X and Z stabilizers do not commute")
        k_expected = n - gf2.rank(hx) - gf2.rank(hz)
        if lx.shape[0] != k_expected or lz.shape[0] != k_expected:
            raise CodeError(f"{self.name}: expected k={k_expected}, got {lx.shape[0]}/{lz.shape[0]} logicals")
        if hz.size and gf2.matmul(lx, hz.T).any():
            raise CodeError(f"{self.name}: logical X anticommutes with a Z stabilizer")
        if hx.size and gf2.matmul(lz, hx.T).any():
            raise CodeError(f"{self.name}: logical Z anticommutes with an X stabilizer")
        if k_expected and not np.array_equal(gf2.matmul(lx, lz.T), np.eye(k_expected, dtype=np.uint8)):
            raise CodeError(f"{self.name}: logicals are not in canonical form (lx lz^T != I)")
        if self.schedule is not None:
            s = self.schedule
            if len(s.x_steps) != self.num_x_checks or len(s.z_steps) != self.num_z_checks:
                raise CodeError(f"{self.name}: schedule has wrong number of checks")
            depth = s.depth
            for kind, steps, h in (("X", s.x_steps, hx), ("Z", s.z_steps, hz)):
                for c, row in enumerate(steps):
                    if len(row) != depth:
                        raise CodeError(f"{self.name}: ragged schedule")
                    touched = sorted(q for q in row if q >= 0)
                    if touched != sorted(np.nonzero(h[c])[0].tolist()):
                        raise CodeError(f"{self.name}: schedule for {kind} check {c} does not match support")
            for t in range(depth):
                used = [row[t] for row in s.x_steps + s.z_steps if row[t] >= 0]
                if len(used) != len(set(used)):
                    raise CodeError(f"{self.name}: data qubit used twice in schedule step {t}")
        return self

    def with_distance(self, d: int) -> "CSSCode":
        return dataclasses.replace(self, distance=d)

    def to_dict(self) -> dict:
        return {
            "name": self.name, "family": self.family, "n": self.n, "k": self.k,
            "distance": self.distance, "params": dict(self.params),
            "hx": self.hx.tolist(), "hz": self.hz.tolist(),
            "logical_x": self.logical_x.tolist(), "logical_z": self.logical_z.tolist(),
        }


def compute_logicals(hx: np.ndarray, hz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Canonical logical operators (lx, lz) with lx lz^T = I."""
    hx = gf2.as_gf2(hx)
    hz = gf2.as_gf2(hz)
    n = hx.shape[1] if hx.size else hz.shape[1]
    k = n - gf2.rank(hx) - gf2.rank(hz)
    if k <= 0:
        return np.zeros((0, n), np.uint8), np.zeros((0, n), np.uint8)
    ker_hz = gf2.nullspace(hz) if hz.size else np.eye(n, dtype=np.uint8)
    ker_hx = gf2.nullspace(hx) if hx.size else np.eye(n, dtype=np.uint8)
    base_x = gf2.row_basis(hx) if hx.size else np.zeros((0, n), np.uint8)
    base_z = gf2.row_basis(hz) if hz.size else np.zeros((0, n), np.uint8)
    lx = gf2.independent_extension(base_x, ker_hz, k)
    lz = gf2.independent_extension(base_z, ker_hx, k)
    if lx.shape[0] != k or lz.shape[0] != k:
        raise CodeError("failed to find a full set of logical operators")
    m = gf2.matmul(lx, lz.T)
    lz = gf2.matmul(gf2.inverse(m).T, lz)
    return lx, lz


def make_css_code(name: str, hx, hz, logical_x=None, logical_z=None, **kw) -> CSSCode:
    hx = gf2.as_gf2(hx).astype(np.uint8)
    hz = gf2.as_gf2(hz).astype(np.uint8)
    if hx.ndim != 2 or hz.ndim != 2:
        raise CodeError("parity check matrices must be 2-D (use shape (0, n) for none)")
    if logical_x is None or logical_z is None:
        lx, lz = compute_logicals(hx, hz)
    else:
        lx = gf2.as_gf2(logical_x).astype(np.uint8)
        lz = gf2.as_gf2(logical_z).astype(np.uint8)
    code = CSSCode(name=name, hx=hx, hz=hz, logical_x=lx, logical_z=lz, **kw)
    return code.validate()

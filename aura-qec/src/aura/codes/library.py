"""
Code library: repetition, rotated surface and bivariate-bicycle (BB) codes.

BB codes follow Bravyi et al., "High-threshold and low-overhead fault-tolerant
quantum memory", Nature 627, 778 (2024):  x = S_l (x) I_m,  y = I_l (x) S_m,
H_X = [A | B],  H_Z = [B^T | A^T].
"""

from __future__ import annotations

import re
from typing import Iterable, Optional

import numpy as np

from ..core.errors import CodeError
from . import gf2
from .css import CheckSchedule, CSSCode, make_css_code


# --------------------------------------------------------------------------- repetition
def repetition_code(d: int) -> CSSCode:
    """Bit-flip repetition code: Z_i Z_{i+1} checks, protects against X errors."""
    if d < 2:
        raise CodeError("repetition code needs d >= 2")
    hz = np.zeros((d - 1, d), dtype=np.uint8)
    for i in range(d - 1):
        hz[i, i] = hz[i, i + 1] = 1
    hx = np.zeros((0, d), dtype=np.uint8)
    lx = np.ones((1, d), dtype=np.uint8)
    lz = np.zeros((1, d), dtype=np.uint8)
    lz[0, 0] = 1
    coords = np.array([[2 * i, 0] for i in range(d)], dtype=float)
    zc = np.array([[2 * i + 1, 0] for i in range(d - 1)], dtype=float)
    steps_z = tuple((i, i + 1) for i in range(d - 1))
    sched = CheckSchedule(x_steps=(), z_steps=steps_z)
    return make_css_code(f"repetition-{d}", hx, hz, lx, lz, distance=d, data_coords=coords,
                         x_check_coords=np.zeros((0, 2)), z_check_coords=zc, schedule=sched,
                         family="repetition", params={"d": d})


# --------------------------------------------------------------------------- rotated surface
# CNOT orders (offsets from the check to the data qubit) chosen so that X and Z
# checks can be measured simultaneously and hook errors run perpendicular to the
# logical operator of the same type.  Validated in tests by noiseless
# determinism and by the circuit-level distance search.
_X_ORDER = [(1, 1), (-1, 1), (1, -1), (-1, -1)]
_Z_ORDER = [(1, 1), (1, -1), (-1, 1), (-1, -1)]


def rotated_surface_code(d: int) -> CSSCode:
    if d < 3 or d % 2 == 0:
        raise CodeError("rotated surface code needs odd d >= 3")
    data_index = {}
    coords = []
    for i in range(d):
        for j in range(d):
            data_index[(2 * i + 1, 2 * j + 1)] = len(coords)
            coords.append((2 * i + 1, 2 * j + 1))
    xs, zs = [], []
    for a in range(d + 1):
        for b in range(d + 1):
            is_x = (a + b) % 2 == 0
            bulk = 1 <= a <= d - 1 and 1 <= b <= d - 1
            tb = (b in (0, d)) and 1 <= a <= d - 1
            lr = (a in (0, d)) and 1 <= b <= d - 1
            if bulk or (tb and is_x) or (lr and not is_x):
                (xs if is_x else zs).append((2 * a, 2 * b))
    n = d * d

    def build(checks, order):
        h = np.zeros((len(checks), n), dtype=np.uint8)
        steps = []
        for c, (cx, cy) in enumerate(checks):
            row = []
            for dx, dy in order:
                q = data_index.get((cx + dx, cy + dy), -1)
                row.append(q)
                if q >= 0:
                    h[c, q] = 1
            steps.append(tuple(row))
        return h, tuple(steps)

    hx, x_steps = build(xs, _X_ORDER)
    hz, z_steps = build(zs, _Z_ORDER)

    # minimum-weight logicals: a full row / column of data qubits
    def find_line(h_commute, h_stab):
        lines = []
        for i in range(d):
            lines.append([data_index[(2 * i + 1, 2 * j + 1)] for j in range(d)])
            lines.append([data_index[(2 * j + 1, 2 * i + 1)] for j in range(d)])
        for ln in lines:
            v = np.zeros(n, dtype=np.uint8)
            v[ln] = 1
            if not gf2.matmul(h_commute, v[:, None]).any() and not gf2.in_rowspace(h_stab, v):
                return v
        raise CodeError("no weight-d logical found (layout bug)")

    lx = find_line(hz, hx)[None, :]
    lz = find_line(hx, hz)[None, :]
    if int(gf2.matmul(lx, lz.T)[0, 0]) != 1:
        raise CodeError("surface code logicals do not anticommute")
    return make_css_code(
        f"rotated-surface-{d}", hx, hz, lx, lz, distance=d,
        data_coords=np.array(coords, dtype=float),
        x_check_coords=np.array(xs, dtype=float), z_check_coords=np.array(zs, dtype=float),
        schedule=CheckSchedule(x_steps=x_steps, z_steps=z_steps),
        family="rotated_surface", params={"d": d})


# --------------------------------------------------------------------------- bivariate bicycle
Monomial = tuple[int, int]   # (power of x, power of y)


def _shift(size: int, power: int) -> np.ndarray:
    return np.roll(np.eye(size, dtype=np.uint8), power % size, axis=1)


def _poly_matrix(l: int, m: int, terms: Iterable[Monomial]) -> np.ndarray:
    out = np.zeros((l * m, l * m), dtype=np.uint8)
    for px, py in terms:
        out ^= np.kron(_shift(l, px), _shift(m, py))
    return out


BB_PRESETS: dict[str, dict] = {
    # name: l, m, A terms, B terms, known distance (Bravyi et al. 2024, Table 3)
    "72": dict(l=6, m=6, A=[(3, 0), (0, 1), (0, 2)], B=[(0, 3), (1, 0), (2, 0)], d=6),
    "90": dict(l=15, m=3, A=[(9, 0), (0, 1), (0, 2)], B=[(0, 0), (2, 0), (7, 0)], d=10),
    "108": dict(l=9, m=6, A=[(3, 0), (0, 1), (0, 2)], B=[(0, 3), (1, 0), (2, 0)], d=10),
    "144": dict(l=12, m=6, A=[(3, 0), (0, 1), (0, 2)], B=[(0, 3), (1, 0), (2, 0)], d=12),
    "288": dict(l=12, m=12, A=[(3, 0), (0, 2), (0, 7)], B=[(0, 3), (1, 0), (2, 0)], d=18),
}
BB_PRESETS["gross"] = BB_PRESETS["144"]


def bivariate_bicycle_code(l: int, m: int, a_terms: list[Monomial], b_terms: list[Monomial],
                           distance: Optional[int] = None, name: Optional[str] = None) -> CSSCode:
    A = _poly_matrix(l, m, a_terms)
    B = _poly_matrix(l, m, b_terms)
    hx = np.hstack([A, B])
    hz = np.hstack([B.T, A.T])
    half = l * m
    coords = np.zeros((2 * half, 2))
    xc = np.zeros((half, 2))
    zc = np.zeros((half, 2))
    for i in range(l):
        for j in range(m):
            idx = i * m + j
            coords[idx] = (2 * i + 1, 2 * j)          # "L" data
            coords[half + idx] = (2 * i, 2 * j + 1)   # "R" data
            xc[idx] = (2 * i, 2 * j)
            zc[idx] = (2 * i + 1, 2 * j + 1)
    n = 2 * half
    code = make_css_code(name or f"bb-{n}", hx, hz, distance=distance, data_coords=coords,
                         x_check_coords=xc, z_check_coords=zc, family="bivariate_bicycle",
                         params={"l": l, "m": m, "A": [list(t) for t in a_terms],
                                 "B": [list(t) for t in b_terms]})
    return code


def bb_code(preset: str) -> CSSCode:
    p = BB_PRESETS.get(str(preset))
    if p is None:
        raise CodeError(f"unknown BB preset {preset!r}; choose from {sorted(BB_PRESETS)}")
    n = 2 * p["l"] * p["m"]
    code = bivariate_bicycle_code(p["l"], p["m"], p["A"], p["B"], distance=p["d"],
                                  name=f"bb-{n}" if preset != "gross" else "bb-gross-144")
    return code


# --------------------------------------------------------------------------- registry
def get_code(spec: str) -> CSSCode:
    """Parse a code spec: 'rep:5', 'surface:5', 'bb:72', 'bb:gross'."""
    mt = re.fullmatch(r"\s*([a-z_]+)\s*:\s*([A-Za-z0-9]+)\s*", spec)
    if not mt:
        raise CodeError(f"bad code spec {spec!r} (expected family:param)")
    fam, arg = mt.group(1), mt.group(2)
    if fam in ("rep", "repetition"):
        return repetition_code(int(arg))
    if fam in ("surface", "rotated_surface", "rsc"):
        return rotated_surface_code(int(arg))
    if fam in ("bb", "bivariate_bicycle"):
        return bb_code(arg)
    raise CodeError(f"unknown code family {fam!r}")

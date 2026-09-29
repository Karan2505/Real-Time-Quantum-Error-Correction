"""Linear algebra over GF(2) on numpy uint8 arrays (values 0/1)."""

from __future__ import annotations

import numpy as np


def as_gf2(m) -> np.ndarray:
    a = np.asarray(m, dtype=np.uint8) & 1
    return a


def row_reduce(m: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """Reduced row-echelon form.  Returns (R, pivot_columns)."""
    a = as_gf2(m).copy()
    if a.ndim != 2:
        raise ValueError("expected a 2-D matrix")
    rows, cols = a.shape
    pivots: list[int] = []
    r = 0
    for c in range(cols):
        if r >= rows:
            break
        nz = np.nonzero(a[r:, c])[0]
        if nz.size == 0:
            continue
        p = r + int(nz[0])
        if p != r:
            a[[r, p]] = a[[p, r]]
        mask = a[:, c].astype(bool)
        mask[r] = False
        a[mask] ^= a[r]
        pivots.append(c)
        r += 1
    return a, pivots


def rank(m: np.ndarray) -> int:
    a = as_gf2(m)
    if a.size == 0:
        return 0
    return len(row_reduce(a)[1])


def nullspace(m: np.ndarray) -> np.ndarray:
    """Basis (as rows) of {v : m @ v = 0 mod 2}."""
    a = as_gf2(m)
    cols = a.shape[1]
    if a.shape[0] == 0:
        return np.eye(cols, dtype=np.uint8)
    r, piv = row_reduce(a)
    free = [c for c in range(cols) if c not in set(piv)]
    basis = []
    for f in free:
        v = np.zeros(cols, dtype=np.uint8)
        v[f] = 1
        for i, pc in enumerate(piv):
            if r[i, f]:
                v[pc] = 1
        basis.append(v)
    if not basis:
        return np.zeros((0, cols), dtype=np.uint8)
    return np.array(basis, dtype=np.uint8)


def row_basis(m: np.ndarray) -> np.ndarray:
    a = as_gf2(m)
    if a.shape[0] == 0:
        return a.copy()
    r, piv = row_reduce(a)
    return r[: len(piv)].copy()


def in_rowspace(m: np.ndarray, v: np.ndarray) -> bool:
    a = as_gf2(m)
    if a.shape[0] == 0:
        return not as_gf2(v).any()
    return rank(np.vstack([a, as_gf2(v)[None, :]])) == rank(a)


def inverse(m: np.ndarray) -> np.ndarray:
    """Inverse of a square invertible GF(2) matrix."""
    a = as_gf2(m)
    n = a.shape[0]
    if a.shape != (n, n):
        raise ValueError("matrix must be square")
    aug = np.hstack([a, np.eye(n, dtype=np.uint8)])
    r, piv = row_reduce(aug)
    if piv[:n] != list(range(n)):
        raise ValueError("matrix is singular over GF(2)")
    return r[:, n:].copy()


def matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (as_gf2(a).astype(np.int64) @ as_gf2(b).astype(np.int64) % 2).astype(np.uint8)


def independent_extension(base: np.ndarray, candidates: np.ndarray, k: int) -> np.ndarray:
    """Pick up to ``k`` candidate rows that are linearly independent of ``base``
    and of each other.  Returns them as rows."""
    cur = as_gf2(base)
    cur_rank = rank(cur) if cur.shape[0] else 0
    picked = []
    for v in as_gf2(candidates):
        test = np.vstack([cur, v[None, :]]) if cur.shape[0] else v[None, :]
        rr = rank(test)
        if rr > cur_rank:
            picked.append(v.copy())
            cur, cur_rank = test, rr
            if len(picked) == k:
                break
    if not picked:
        return np.zeros((0, as_gf2(candidates).shape[1]), dtype=np.uint8)
    return np.array(picked, dtype=np.uint8)

"""
Pauli channels: construction, sampling and conversion to independent
error mechanisms.

Conventions
-----------
Single-qubit Paulis are indexed I=0, X=1, Y=2, Z=3 and map to frame bits
(x, z): I=(0,0) X=(1,0) Y=(1,1) Z=(0,1).  A k-qubit channel is a probability
vector of length 4**k over tensor products, index = sum(p_j * 4**(k-1-j))
(first qubit most significant).  Entry 0 is the identity.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Sequence

import numpy as np

PAULI_X_BIT = np.array([0, 1, 1, 0], dtype=np.uint8)
PAULI_Z_BIT = np.array([0, 0, 1, 1], dtype=np.uint8)
PAULI_NAMES = "IXYZ"


def pauli_bits(index: int, k: int) -> tuple[np.ndarray, np.ndarray]:
    """(x_bits, z_bits) of length k for a k-qubit Pauli index."""
    xs = np.zeros(k, dtype=np.uint8)
    zs = np.zeros(k, dtype=np.uint8)
    for j in range(k):
        p = (index // 4 ** (k - 1 - j)) % 4
        xs[j] = PAULI_X_BIT[p]
        zs[j] = PAULI_Z_BIT[p]
    return xs, zs


def pauli_name(index: int, k: int) -> str:
    return "".join(PAULI_NAMES[(index // 4 ** (k - 1 - j)) % 4] for j in range(k))


def _check(probs: np.ndarray) -> np.ndarray:
    p = np.asarray(probs, dtype=float)
    if np.any(p < -1e-15):
        raise ValueError(f"negative probability in Pauli channel: {p}")
    p = np.clip(p, 0.0, None)
    s = p[1:].sum()
    if s > 1.0 + 1e-12:
        raise ValueError(f"Pauli channel error probabilities sum to {s} > 1")
    p[0] = max(0.0, 1.0 - s)
    return p


def pauli_channel_1(px: float, py: float, pz: float) -> np.ndarray:
    return _check(np.array([0.0, px, py, pz]))


def depolarize1(p: float) -> np.ndarray:
    return _check(np.array([0.0, p / 3, p / 3, p / 3]))


def depolarize2(p: float) -> np.ndarray:
    v = np.full(16, p / 15.0)
    v[0] = 0.0
    return _check(v)


def compose(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Channel equivalent to applying ``a`` then ``b`` independently (same qubits)."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if a.shape != b.shape:
        raise ValueError("channels act on different numbers of qubits")
    k = int(round(math.log(len(a), 4)))
    table = _product_table(k)
    out = np.zeros_like(a)
    nz_a = np.nonzero(a)[0]
    nz_b = np.nonzero(b)[0]
    for i in nz_a:
        for j in nz_b:
            out[table[i, j]] += a[i] * b[j]
    return _check(out)


def tensor(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Independent channels on disjoint qubit sets -> joint channel (a first)."""
    return _check(np.kron(np.asarray(a, float), np.asarray(b, float)))


@lru_cache(maxsize=None)
def _product_table(k: int) -> np.ndarray:
    """table[i, j] = index of Pauli product (up to phase)."""
    size = 4 ** k
    xz = [pauli_bits(i, k) for i in range(size)]
    lookup = {}
    for i, (x, z) in enumerate(xz):
        lookup[(x.tobytes(), z.tobytes())] = i
    t = np.zeros((size, size), dtype=np.int64)
    for i in range(size):
        for j in range(size):
            x = xz[i][0] ^ xz[j][0]
            z = xz[i][1] ^ xz[j][1]
            t[i, j] = lookup[(x.tobytes(), z.tobytes())]
    return t


@lru_cache(maxsize=None)
def _symplectic_signs(k: int) -> np.ndarray:
    """S[s, P] = (-1)^{<s,P>} with the symplectic inner product."""
    size = 4 ** k
    xz = [pauli_bits(i, k) for i in range(size)]
    m = np.zeros((size, size))
    for s in range(size):
        xs, zs = xz[s]
        for p in range(size):
            xp, zp = xz[p]
            ip = (int(xs @ zp) + int(zs @ xp)) % 2
            m[s, p] = -1.0 if ip else 1.0
    return m


def to_independent(probs: Sequence[float], strict: bool = False) -> np.ndarray:
    """Convert a (disjoint-outcome) Pauli channel into independent mechanisms.

    Returns q (same indexing, q[0]=0) such that applying each non-identity
    Pauli P independently with probability q[P] reproduces exactly the input
    channel.  Uses the factorisation of the channel's characteristic function
    f(s) = sum_P p_P (-1)^{<s,P>} = prod_P (1 - 2 q_P [<s,P>=1]).
    Requires f(s) > 0 for all s (true for any physically small error rate);
    otherwise raises if ``strict`` else falls back to q = p.
    """
    p = _check(np.asarray(probs, float))
    size = len(p)
    k = int(round(math.log(size, 4)))
    signs = _symplectic_signs(k)
    f = signs @ p
    if np.any(f <= 0):
        if strict:
            raise ValueError("channel too noisy for an independent decomposition")
        # Very noisy channel (e.g. an uncalibrated device): fall back to the
        # first-order approximation "disjoint rates == independent rates".
        q = p.copy()
        q[0] = 0.0
        return np.clip(q, 0.0, 0.5)
    logf = np.log(f)
    # log f(s) = sum_P [<s,P>=1] L_P, with L_P = log(1 - 2 q_P);  A[s,P] = (1 - sign)/2
    a = (1.0 - signs) / 2.0
    a_red = a[1:, 1:]
    lvec = np.linalg.solve(a_red, logf[1:])
    q = np.zeros(size)
    q[1:] = (1.0 - np.exp(lvec)) / 2.0
    q[np.abs(q) < 1e-18] = 0.0
    if np.any(q < -1e-12):
        raise ValueError("channel has no independent decomposition with non-negative rates")
    return np.clip(q, 0.0, None)


def pta_idle(t1_ps: float, t2_ps: float, dt_ps: float) -> np.ndarray:
    """Pauli-twirl approximation of amplitude damping (T1) + pure dephasing
    (T2 <= 2 T1) over an idle interval.

        px = py = (1 - exp(-t/T1)) / 4
        pz = (1 - exp(-t/T2)) / 2 - (1 - exp(-t/T1)) / 4
    """
    if dt_ps <= 0:
        return np.array([1.0, 0.0, 0.0, 0.0])
    if t1_ps <= 0 or t2_ps <= 0:
        raise ValueError("T1/T2 must be positive")
    t2 = min(t2_ps, 2.0 * t1_ps)
    a = 1.0 - math.exp(-dt_ps / t1_ps)
    px = a / 4.0
    pz = (1.0 - math.exp(-dt_ps / t2)) / 2.0 - a / 4.0
    return _check(np.array([0.0, px, px, max(pz, 0.0)]))


def coherent_rotation_twirl(angle_rad: float) -> float:
    """Probability of the Pauli error after twirling a small coherent rotation
    exp(-i angle/2 P): p = sin^2(angle/2)."""
    return math.sin(angle_rad / 2.0) ** 2


def sample_paulis(probs: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    """Sample ``count`` Pauli indices from a channel (vectorised)."""
    p = _check(probs)
    if p[0] >= 1.0:
        return np.zeros(count, dtype=np.int64)
    return rng.choice(len(p), size=count, p=p / p.sum())

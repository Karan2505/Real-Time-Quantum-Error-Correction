"""
Stabilizer simulation engines.

* :class:`Tableau` -- exact Aaronson-Gottesman (CHP) stabilizer tableau with
  destabilizers.  Row operations are vectorised with numpy.  Used for
  reference samples, validation and small exact simulations.

* :class:`FrameSimulator` -- batched Pauli-frame simulator.  Tracks, for B
  shots at once, the Pauli difference between each noisy shot and a noiseless
  reference execution.  Exact for Clifford circuits with Pauli noise
  (same technique as Stim's frame simulator): measurement result =
  reference XOR frame_x, and after every measurement / reset the frame's Z
  component is randomised (it is a stabilizer of the post-measurement state).

Supported gate names (both engines):
  I, X, Y, Z, H, S, S_DAG, SX, SX_DAG, CX (control, target), CZ
plus measure_z / reset_z.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

SINGLE_QUBIT_GATES = ("I", "X", "Y", "Z", "H", "S", "S_DAG", "SX", "SX_DAG")
TWO_QUBIT_GATES = ("CX", "CZ")


def _as_idx(qs) -> np.ndarray:
    return np.atleast_1d(np.asarray(qs, dtype=np.int64))


class Tableau:
    """CHP tableau over n qubits initialised to |0...0>.

    Rows 0..n-1 are destabilizers, rows n..2n-1 stabilizers.  Each row is a
    Hermitian Pauli (-1)^r i^{x.z} X^x Z^z.
    """

    def __init__(self, n: int, rng: Optional[np.random.Generator] = None):
        if n < 1:
            raise ValueError("need at least one qubit")
        self.n = n
        self.x = np.zeros((2 * n, n), dtype=np.uint8)
        self.z = np.zeros((2 * n, n), dtype=np.uint8)
        self.r = np.zeros(2 * n, dtype=np.uint8)
        self.x[np.arange(n), np.arange(n)] = 1          # destabilizers X_i
        self.z[n + np.arange(n), np.arange(n)] = 1      # stabilizers Z_i
        self.rng = rng if rng is not None else np.random.default_rng(0)

    def copy(self) -> "Tableau":
        t = Tableau.__new__(Tableau)
        t.n = self.n
        t.x = self.x.copy()
        t.z = self.z.copy()
        t.r = self.r.copy()
        t.rng = self.rng
        return t

    # ------------------------------------------------------------------ gates
    def h(self, qs) -> None:
        q = _as_idx(qs)
        self.r ^= np.bitwise_xor.reduce(self.x[:, q] & self.z[:, q], axis=1)
        xq = self.x[:, q].copy()
        self.x[:, q] = self.z[:, q]
        self.z[:, q] = xq

    def s(self, qs) -> None:
        q = _as_idx(qs)
        self.r ^= np.bitwise_xor.reduce(self.x[:, q] & self.z[:, q], axis=1)
        self.z[:, q] ^= self.x[:, q]

    def s_dag(self, qs) -> None:
        # S_dag = S^3
        for _ in range(3):
            self.s(qs)

    def pauli_x(self, qs) -> None:
        q = _as_idx(qs)
        self.r ^= np.bitwise_xor.reduce(self.z[:, q], axis=1)

    def pauli_z(self, qs) -> None:
        q = _as_idx(qs)
        self.r ^= np.bitwise_xor.reduce(self.x[:, q], axis=1)

    def pauli_y(self, qs) -> None:
        q = _as_idx(qs)
        self.r ^= np.bitwise_xor.reduce(self.x[:, q] ^ self.z[:, q], axis=1)

    def sx(self, qs) -> None:
        # SX = H S H  (up to global phase)
        self.h(qs)
        self.s(qs)
        self.h(qs)

    def sx_dag(self, qs) -> None:
        self.h(qs)
        self.s_dag(qs)
        self.h(qs)

    def cx(self, controls, targets) -> None:
        c = _as_idx(controls)
        t = _as_idx(targets)
        if c.shape != t.shape:
            raise ValueError("controls/targets length mismatch")
        if len(set(c.tolist()) | set(t.tolist())) != 2 * len(c):
            raise ValueError("CX pairs must be disjoint")
        xc, zc, xt, zt = self.x[:, c], self.z[:, c], self.x[:, t], self.z[:, t]
        self.r ^= np.bitwise_xor.reduce(xc & zt & (xt ^ zc ^ 1), axis=1)
        self.x[:, t] = xt ^ xc
        self.z[:, c] = zc ^ zt

    def cz(self, a, b) -> None:
        self.h(b)
        self.cx(a, b)
        self.h(b)

    def apply(self, name: str, qubits: Sequence[int]) -> None:
        q = list(qubits)
        if name == "I":
            return
        if name in TWO_QUBIT_GATES:
            if len(q) % 2:
                raise ValueError(f"{name} needs pairs of qubits")
            (self.cx if name == "CX" else self.cz)(q[0::2], q[1::2])
            return
        fn = {"X": self.pauli_x, "Y": self.pauli_y, "Z": self.pauli_z, "H": self.h, "S": self.s,
              "S_DAG": self.s_dag, "SX": self.sx, "SX_DAG": self.sx_dag}.get(name)
        if fn is None:
            raise ValueError(f"unknown gate {name}")
        fn(q)

    # ------------------------------------------------------- row arithmetic
    @staticmethod
    def _product_sign(xs: np.ndarray, zs: np.ndarray, rs: np.ndarray) -> tuple[np.ndarray, np.ndarray, int, bool]:
        """Ordered product of rows (k x n).  Returns (x, z, sign_bit, hermitian)."""
        xi = xs.astype(np.int64)
        zi = zs.astype(np.int64)
        suffix = np.cumsum(xi[::-1], axis=0)[::-1] - xi          # sum_{j>i} x_j
        cross = int((zi * suffix).sum())
        x = (xi.sum(axis=0) % 2).astype(np.uint8)
        z = (zi.sum(axis=0) % 2).astype(np.uint8)
        e = (int((xi * zi).sum()) + 2 * cross - int(x.astype(np.int64) @ z.astype(np.int64))) % 4
        sign = (int(rs.sum()) + e // 2) % 2
        return x, z, sign, e % 2 == 0

    def _multiply_rows_by(self, targets: np.ndarray, src: int) -> None:
        """row_t <- row_t * row_src for every t in targets (vectorised)."""
        if targets.size == 0:
            return
        xa, za = self.x[targets].astype(np.int64), self.z[targets].astype(np.int64)
        xb, zb = self.x[src].astype(np.int64), self.z[src].astype(np.int64)
        x = xa ^ xb
        z = za ^ zb
        e = ((xa * za).sum(1) + int(xb @ zb) + 2 * (za @ xb) - (x * z).sum(1)) % 4
        self.r[targets] = (self.r[targets] + self.r[src] + e // 2) % 2
        self.x[targets] = x.astype(np.uint8)
        self.z[targets] = z.astype(np.uint8)

    # ------------------------------------------------------------ measurement
    def peek_z(self, q: int) -> int:
        """+1 / -1 if Z_q is deterministic, 0 if the outcome is random."""
        n = self.n
        if self.x[n:, q].any():
            return 0
        rows = n + np.nonzero(self.x[:n, q])[0]
        if rows.size == 0:
            return +1
        _, _, sign, _ = self._product_sign(self.x[rows], self.z[rows], self.r[rows])
        return -1 if sign else +1

    def measure_z(self, q: int, forced: Optional[int] = None) -> tuple[int, bool]:
        """Projective Z measurement.  Returns (bit, was_random).

        ``forced`` fixes the outcome of a random measurement (post-selection);
        forcing a deterministic measurement to the wrong value raises.
        """
        n = self.n
        anti = np.nonzero(self.x[n:, q])[0]
        if anti.size:
            p = n + int(anti[0])
            others = np.nonzero(self.x[:, q])[0]
            others = others[others != p]
            self._multiply_rows_by(others, p)
            d = p - n
            self.x[d] = self.x[p]
            self.z[d] = self.z[p]
            self.r[d] = self.r[p]
            self.x[p] = 0
            self.z[p] = 0
            self.z[p, q] = 1
            bit = int(self.rng.integers(2)) if forced is None else int(forced) & 1
            self.r[p] = bit
            return bit, True
        rows = n + np.nonzero(self.x[:n, q])[0]
        if rows.size == 0:
            bit = 0
        else:
            _, _, bit, herm = self._product_sign(self.x[rows], self.z[rows], self.r[rows])
            if not herm:  # pragma: no cover - would indicate a corrupted tableau
                raise RuntimeError("tableau corrupted: non-Hermitian stabilizer product")
        if forced is not None and (int(forced) & 1) != bit:
            raise ValueError(f"cannot force deterministic measurement of qubit {q} to {forced}")
        return bit, False

    def reset_z(self, q: int) -> None:
        bit, _ = self.measure_z(q)
        if bit:
            self.pauli_x([q])

    def stabilizer_strings(self) -> list[str]:
        """Human-readable stabilizer generators (for debugging / tests)."""
        out = []
        n = self.n
        for i in range(n, 2 * n):
            s = "-" if self.r[i] else "+"
            for q in range(n):
                s += ".XZY"[self.x[i, q] + 2 * self.z[i, q]]
            out.append(s)
        return out


class FrameSimulator:
    """Batched Pauli frames: ``fx``, ``fz`` have shape (n_qubits, batch)."""

    def __init__(self, n: int, batch: int, rng: np.random.Generator):
        if n < 1 or batch < 1:
            raise ValueError("need n >= 1 and batch >= 1")
        self.n = n
        self.batch = batch
        self.rng = rng
        self.fx = np.zeros((n, batch), dtype=bool)
        self.fz = rng.integers(0, 2, size=(n, batch), dtype=np.uint8).astype(bool)

    # gates (Pauli gates do not change frames: signs live in the reference)
    def h(self, qs) -> None:
        q = _as_idx(qs)
        self.fx[q], self.fz[q] = self.fz[q].copy(), self.fx[q].copy()

    def s(self, qs) -> None:
        q = _as_idx(qs)
        self.fz[q] ^= self.fx[q]

    s_dag = s

    def sx(self, qs) -> None:
        q = _as_idx(qs)
        self.fx[q] ^= self.fz[q]

    sx_dag = sx

    def cx(self, controls, targets) -> None:
        c = _as_idx(controls)
        t = _as_idx(targets)
        self.fx[t] ^= self.fx[c]
        self.fz[c] ^= self.fz[t]

    def cz(self, a, b) -> None:
        a = _as_idx(a)
        b = _as_idx(b)
        xa = self.fx[a].copy()
        self.fz[a] ^= self.fx[b]
        self.fz[b] ^= xa

    def apply(self, name: str, qubits: Sequence[int]) -> None:
        q = list(qubits)
        if name in ("I", "X", "Y", "Z"):
            return
        if name == "CX":
            self.cx(q[0::2], q[1::2])
        elif name == "CZ":
            self.cz(q[0::2], q[1::2])
        elif name == "H":
            self.h(q)
        elif name in ("S", "S_DAG"):
            self.s(q)
        elif name in ("SX", "SX_DAG"):
            self.sx(q)
        else:
            raise ValueError(f"unknown gate {name}")

    # noise
    def flip_x(self, q: int, mask: np.ndarray) -> None:
        self.fx[q] ^= mask

    def flip_z(self, q: int, mask: np.ndarray) -> None:
        self.fz[q] ^= mask

    def apply_pauli_indices(self, qubits: Sequence[int], idx: np.ndarray) -> None:
        """Apply sampled k-qubit Pauli indices (shape (batch,)) to the frames."""
        k = len(qubits)
        for j, q in enumerate(qubits):
            p = (idx // 4 ** (k - 1 - j)) % 4
            self.fx[q] ^= (p == 1) | (p == 2)
            self.fz[q] ^= (p == 2) | (p == 3)

    # measurement / reset
    def measure_z(self, q: int) -> np.ndarray:
        flips = self.fx[q].copy()
        self.fz[q] = self.rng.integers(0, 2, size=self.batch, dtype=np.uint8).astype(bool)
        return flips

    def reset_z(self, q: int) -> None:
        self.fx[q] = False
        self.fz[q] = self.rng.integers(0, 2, size=self.batch, dtype=np.uint8).astype(bool)

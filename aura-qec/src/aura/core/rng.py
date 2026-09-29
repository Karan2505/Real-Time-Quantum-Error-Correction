"""
Reproducible, order-independent random streams.

``RngFactory(seed).stream("qpu/readout")`` always returns a generator with the
same state for the same (seed, name), regardless of how many other streams
were created before.  This keeps runs reproducible even when components are
added, removed or constructed in a different order.
"""

from __future__ import annotations

import hashlib

import numpy as np


def _name_key(name: str) -> tuple[int, ...]:
    d = hashlib.blake2b(name.encode("utf-8"), digest_size=16).digest()
    return tuple(int.from_bytes(d[i:i + 4], "little") for i in range(0, 16, 4))


class RngFactory:
    def __init__(self, seed: int = 0):
        if seed < 0:
            raise ValueError("seed must be non-negative")
        self.seed = int(seed)

    def stream(self, name: str) -> np.random.Generator:
        ss = np.random.SeedSequence(entropy=self.seed, spawn_key=_name_key(name))
        return np.random.Generator(np.random.PCG64(ss))

    def child(self, name: str) -> "RngFactory":
        """A derived factory (e.g. per run) whose streams are independent of the parent's."""
        key = _name_key("child:" + name)
        return RngFactory(int(np.random.SeedSequence(entropy=self.seed, spawn_key=key)
                              .generate_state(1, dtype=np.uint64)[0] >> 1))

"""
CRC-32C (Castagnoli, polynomial 0x1EDC6F41, reflected 0x82F63B78).

Chosen over IEEE CRC-32 for its larger minimum Hamming distance at the frame
lengths AURA-Link uses (HD=6 up to ~5 kbit vs HD=4 for CRC-32/IEEE) and
because it is the CRC implemented in hardware by modern CPUs / NICs.
Check value: crc32c(b"123456789") == 0xE3069283.

Uses the ``crc32c`` / ``google_crc32c`` extension when installed, else a
slicing-by-8 pure-Python implementation.
"""

from __future__ import annotations

_POLY = 0x82F63B78


def _make_tables() -> list[list[int]]:
    t0 = []
    for i in range(256):
        c = i
        for _ in range(8):
            c = (c >> 1) ^ _POLY if c & 1 else c >> 1
        t0.append(c)
    tables = [t0]
    for k in range(1, 8):
        prev = tables[k - 1]
        tables.append([(prev[i] >> 8) ^ t0[prev[i] & 0xFF] for i in range(256)])
    return tables


_T = _make_tables()
_T0, _T1, _T2, _T3, _T4, _T5, _T6, _T7 = _T


def _crc32c_py(data: bytes, crc: int = 0) -> int:
    c = crc ^ 0xFFFFFFFF
    mv = memoryview(data)
    n = len(mv)
    i = 0
    while n - i >= 8:
        lo = c ^ (mv[i] | mv[i + 1] << 8 | mv[i + 2] << 16 | mv[i + 3] << 24)
        c = (_T7[lo & 0xFF] ^ _T6[(lo >> 8) & 0xFF] ^ _T5[(lo >> 16) & 0xFF] ^ _T4[lo >> 24]
             ^ _T3[mv[i + 4]] ^ _T2[mv[i + 5]] ^ _T1[mv[i + 6]] ^ _T0[mv[i + 7]])
        i += 8
    while i < n:
        c = (c >> 8) ^ _T0[(c ^ mv[i]) & 0xFF]
        i += 1
    return c ^ 0xFFFFFFFF


try:  # optional hardware-accelerated backends
    import crc32c as _ext  # type: ignore

    def crc32c(data: bytes, crc: int = 0) -> int:
        return _ext.crc32c(data, crc) & 0xFFFFFFFF
    BACKEND = "crc32c"
except Exception:  # pragma: no cover - depends on environment
    try:
        import google_crc32c as _g  # type: ignore

        def crc32c(data: bytes, crc: int = 0) -> int:
            return _g.extend(crc, data) & 0xFFFFFFFF
        BACKEND = "google_crc32c"
    except Exception:
        crc32c = _crc32c_py
        BACKEND = "python"

assert _crc32c_py(b"123456789") == 0xE3069283, "CRC-32C self-test failed"

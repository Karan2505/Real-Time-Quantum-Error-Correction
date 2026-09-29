"""
Time units.

Every timestamp and duration inside AURA-QEC is an ``int`` number of
**picoseconds**.  Integer picoseconds give exact arithmetic for serial-link
unit intervals (97 ps at 10.3125 Gb/s), FPGA clock periods (3333 ps at
300 MHz) and qubit pulse timing (4 ns sequencer grid) without floating-point
drift over long runs (2**63 ps is roughly 106 days of virtual time).
"""

from __future__ import annotations

PS: int = 1
NS: int = 1_000
US: int = 1_000_000
MS: int = 1_000_000_000
S: int = 1_000_000_000_000


def ns(x: float) -> int:
    """Nanoseconds -> integer picoseconds (rounded to nearest)."""
    return int(round(x * NS))


def us(x: float) -> int:
    """Microseconds -> integer picoseconds."""
    return int(round(x * US))


def ms(x: float) -> int:
    """Milliseconds -> integer picoseconds."""
    return int(round(x * MS))


def to_ns(t_ps: int) -> float:
    return t_ps / NS


def to_us(t_ps: int) -> float:
    return t_ps / US


def period_ps(freq_hz: float) -> int:
    """Clock period in integer picoseconds for a frequency in Hz."""
    if freq_hz <= 0:
        raise ValueError("frequency must be positive")
    return int(round(S / freq_hz))


def ceil_to_grid(t_ps: int, grid_ps: int) -> int:
    """Smallest multiple of ``grid_ps`` that is >= ``t_ps``."""
    if grid_ps <= 0:
        raise ValueError("grid must be positive")
    return -(-t_ps // grid_ps) * grid_ps


def fmt_time(t_ps: int) -> str:
    """Human readable time with an adaptive unit."""
    a = abs(t_ps)
    if a >= S:
        return f"{t_ps / S:.6g} s"
    if a >= MS:
        return f"{t_ps / MS:.6g} ms"
    if a >= US:
        return f"{t_ps / US:.6g} us"
    if a >= NS:
        return f"{t_ps / NS:.6g} ns"
    return f"{t_ps} ps"

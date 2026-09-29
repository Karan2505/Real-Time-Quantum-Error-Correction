"""
Metrics registry: counters, gauges and log-bucketed latency histograms.

Histograms use HDR-style log-linear buckets (``sub_buckets`` per power of two)
so percentiles are accurate to ~1/sub_buckets relative error over a huge
dynamic range (ps .. hours) with constant memory, which is how production
latency telemetry is kept on FPGA/firmware too.  Thread-safe so the web
console can read while a run executes.
"""

from __future__ import annotations

import math
import threading
from typing import Dict, Iterable, Optional


class Counter:
    __slots__ = ("name", "value", "_lock")

    def __init__(self, name: str):
        self.name = name
        self.value = 0
        self._lock = threading.Lock()

    def inc(self, n: int = 1) -> None:
        with self._lock:
            self.value += n


class Gauge:
    __slots__ = ("name", "value", "max_value", "_lock")

    def __init__(self, name: str):
        self.name = name
        self.value = 0.0
        self.max_value = float("-inf")
        self._lock = threading.Lock()

    def set(self, v: float) -> None:
        with self._lock:
            self.value = v
            if v > self.max_value:
                self.max_value = v


class Histogram:
    """Log-linear histogram of non-negative integer values (e.g. latency in ps)."""

    def __init__(self, name: str, sub_buckets: int = 32):
        if sub_buckets < 2 or sub_buckets & (sub_buckets - 1):
            raise ValueError("sub_buckets must be a power of two >= 2")
        self.name = name
        self.sub = sub_buckets
        self._sub_bits = sub_buckets.bit_length() - 1
        self._counts: Dict[int, int] = {}
        self.count = 0
        self.total = 0
        self.min: Optional[int] = None
        self.max: Optional[int] = None
        self._lock = threading.Lock()

    def _index(self, v: int) -> int:
        if v < self.sub:
            return v
        e = v.bit_length() - 1 - self._sub_bits          # exponent above linear range
        m = v >> e                                        # in [sub, 2*sub)
        return (e + 1) * self.sub + (m - self.sub)

    def _bucket_bounds(self, idx: int) -> tuple[int, int]:
        if idx < self.sub:
            return idx, idx
        e = idx // self.sub - 1
        m = idx % self.sub + self.sub
        lo = m << e
        return lo, lo + (1 << e) - 1

    def record(self, v: int, n: int = 1) -> None:
        if v < 0:
            raise ValueError(f"histogram {self.name}: negative value {v}")
        v = int(v)
        i = self._index(v)
        with self._lock:
            self._counts[i] = self._counts.get(i, 0) + n
            self.count += n
            self.total += v * n
            self.min = v if self.min is None else min(self.min, v)
            self.max = v if self.max is None else max(self.max, v)

    def record_many(self, values: Iterable[int]) -> None:
        for v in values:
            self.record(v)

    def percentile(self, p: float) -> Optional[float]:
        """Value at percentile p in [0, 100] (bucket midpoint, clamped to min/max)."""
        with self._lock:
            if self.count == 0:
                return None
            if p <= 0:
                return float(self.min)
            if p >= 100:
                return float(self.max)
            target = math.ceil(self.count * p / 100.0)
            acc = 0
            for idx in sorted(self._counts):
                acc += self._counts[idx]
                if acc >= target:
                    lo, hi = self._bucket_bounds(idx)
                    mid = (lo + hi) / 2.0
                    return float(min(max(mid, self.min), self.max))
            return float(self.max)

    @property
    def mean(self) -> Optional[float]:
        return self.total / self.count if self.count else None

    def snapshot(self) -> dict:
        return {
            "count": self.count,
            "min": self.min,
            "max": self.max,
            "mean": self.mean,
            "p50": self.percentile(50),
            "p90": self.percentile(90),
            "p99": self.percentile(99),
            "p999": self.percentile(99.9),
        }

    def reset(self) -> None:
        with self._lock:
            self._counts.clear()
            self.count = 0
            self.total = 0
            self.min = self.max = None


class MetricsRegistry:
    """Namespace of named metrics; ``snapshot()`` returns a JSON-able dict."""

    def __init__(self, prefix: str = ""):
        self.prefix = prefix
        self._counters: Dict[str, Counter] = {}
        self._gauges: Dict[str, Gauge] = {}
        self._hists: Dict[str, Histogram] = {}
        self._lock = threading.Lock()

    def _full(self, name: str) -> str:
        return f"{self.prefix}.{name}" if self.prefix else name

    def counter(self, name: str) -> Counter:
        with self._lock:
            c = self._counters.get(name)
            if c is None:
                c = self._counters[name] = Counter(self._full(name))
            return c

    def gauge(self, name: str) -> Gauge:
        with self._lock:
            g = self._gauges.get(name)
            if g is None:
                g = self._gauges[name] = Gauge(self._full(name))
            return g

    def histogram(self, name: str, sub_buckets: int = 32) -> Histogram:
        with self._lock:
            h = self._hists.get(name)
            if h is None:
                h = self._hists[name] = Histogram(self._full(name), sub_buckets)
            return h

    def count(self, name: str) -> int:
        c = self._counters.get(name)
        return c.value if c else 0

    def snapshot(self) -> dict:
        with self._lock:
            counters = dict(self._counters)
            gauges = dict(self._gauges)
            hists = dict(self._hists)
        return {
            "counters": {k: v.value for k, v in sorted(counters.items())},
            "gauges": {k: {"value": v.value, "max": (None if v.max_value == float("-inf") else v.max_value)}
                       for k, v in sorted(gauges.items())},
            "histograms": {k: v.snapshot() for k, v in sorted(hists.items())},
        }

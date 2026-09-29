"""
Deterministic discrete-event simulation kernel (virtual time).

All virtual hardware -- qubits, sequencers, readout electronics, serial
links, the ingress gateway -- schedules work on a :class:`Simulator`.
Time is an ``int`` in picoseconds.  Events at the same timestamp run in
(priority, insertion order), so every run with the same seed is
bit-for-bit reproducible.

The same :class:`Scheduler` protocol is also implemented by
:class:`RealtimeScheduler`, which runs callbacks on a background thread
against the host monotonic clock.  Transport components are written against
the protocol and therefore run unchanged in virtual time (digital twin) or
wall-clock time (real sockets).
"""

from __future__ import annotations

import heapq
import itertools
import threading
import time
from typing import Any, Callable, Optional, Protocol

from .errors import AuraError


class Scheduler(Protocol):
    """Minimal timer interface used by every time-aware component."""

    def now_ps(self) -> int: ...

    def call_later(self, delay_ps: int, callback: Callable[..., Any], *args: Any,
                   priority: int = 0) -> "EventHandle": ...

    def cancel(self, handle: "EventHandle") -> None: ...


class EventHandle:
    """Handle returned by ``schedule``; pass to ``cancel`` to revoke the event."""

    __slots__ = ("time", "priority", "seq", "callback", "args", "cancelled")

    def __init__(self, t: int, priority: int, seq: int, callback: Callable[..., Any],
                 args: tuple):
        self.time = t
        self.priority = priority
        self.seq = seq
        self.callback = callback
        self.args = args
        self.cancelled = False

    def __lt__(self, other: "EventHandle") -> bool:
        return (self.time, self.priority, self.seq) < (other.time, other.priority, other.seq)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        name = getattr(self.callback, "__qualname__", repr(self.callback))
        return f"<Event t={self.time}ps prio={self.priority} {name}{' X' if self.cancelled else ''}>"


class SimulationError(AuraError):
    pass


class Simulator:
    """Single-threaded discrete-event kernel with integer picosecond time."""

    def __init__(self, start_ps: int = 0):
        self._now = int(start_ps)
        self._queue: list[EventHandle] = []
        self._seq = itertools.count()
        self._running = False
        self._stop_requested = False
        self.events_processed = 0

    # -- Scheduler protocol -------------------------------------------------
    def now_ps(self) -> int:
        return self._now

    @property
    def now(self) -> int:
        return self._now

    def call_later(self, delay_ps: int, callback: Callable[..., Any], *args: Any,
                   priority: int = 0) -> EventHandle:
        return self.schedule(delay_ps, callback, *args, priority=priority)

    def schedule(self, delay_ps: int, callback: Callable[..., Any], *args: Any,
                 priority: int = 0) -> EventHandle:
        if not isinstance(delay_ps, (int,)) or isinstance(delay_ps, bool):
            raise TypeError(f"delay must be int picoseconds, got {type(delay_ps).__name__}")
        if delay_ps < 0:
            raise SimulationError(f"cannot schedule into the past (delay={delay_ps} ps)")
        return self.schedule_at(self._now + delay_ps, callback, *args, priority=priority)

    def schedule_at(self, t_ps: int, callback: Callable[..., Any], *args: Any,
                    priority: int = 0) -> EventHandle:
        if not isinstance(t_ps, int) or isinstance(t_ps, bool):
            raise TypeError(f"time must be int picoseconds, got {type(t_ps).__name__}")
        if t_ps < self._now:
            raise SimulationError(f"cannot schedule at {t_ps} ps; now is {self._now} ps")
        h = EventHandle(t_ps, priority, next(self._seq), callback, args)
        heapq.heappush(self._queue, h)
        return h

    def cancel(self, handle: Optional[EventHandle]) -> None:
        if handle is not None:
            handle.cancelled = True

    # -- execution ------------------------------------------------------------
    def pending(self) -> int:
        return sum(1 for h in self._queue if not h.cancelled)

    def peek_time(self) -> Optional[int]:
        while self._queue and self._queue[0].cancelled:
            heapq.heappop(self._queue)
        return self._queue[0].time if self._queue else None

    def step(self) -> bool:
        """Run exactly one event.  Returns False if the queue is empty."""
        while self._queue:
            h = heapq.heappop(self._queue)
            if h.cancelled:
                continue
            self._now = h.time
            h.callback(*h.args)
            self.events_processed += 1
            return True
        return False

    def run(self, until_ps: Optional[int] = None, max_events: Optional[int] = None) -> int:
        """Run events until the queue drains, ``until_ps`` is reached, ``stop()``
        is called, or ``max_events`` have executed.  Returns events executed.

        If ``until_ps`` is given, virtual time is advanced to exactly
        ``until_ps`` on return (unless stopped early), even if idle.
        """
        if self._running:
            raise SimulationError("Simulator.run() is not re-entrant")
        self._running = True
        self._stop_requested = False
        n = 0
        try:
            while not self._stop_requested:
                if max_events is not None and n >= max_events:
                    break
                t = self.peek_time()
                if t is None:
                    break
                if until_ps is not None and t > until_ps:
                    break
                self.step()
                n += 1
            if until_ps is not None and not self._stop_requested and self._now < until_ps:
                if max_events is None or n < max_events:
                    self._now = until_ps
        finally:
            self._running = False
        return n

    def run_for(self, duration_ps: int, max_events: Optional[int] = None) -> int:
        return self.run(until_ps=self._now + duration_ps, max_events=max_events)

    def stop(self) -> None:
        """Request ``run`` to return after the current event."""
        self._stop_requested = True


class RealtimeScheduler:
    """Wall-clock implementation of :class:`Scheduler` (background thread).

    Time is read from ``time.monotonic_ns`` (converted to ps).  Callbacks run
    on a single dispatcher thread, preserving the single-threaded callback
    model the transport components assume.
    """

    def __init__(self, name: str = "aura-rt"):
        self._queue: list[EventHandle] = []
        self._seq = itertools.count()
        self._cv = threading.Condition()
        self._closed = False
        self._t0 = time.monotonic_ns()
        self._thread = threading.Thread(target=self._loop, name=name, daemon=True)
        self._thread.start()
        self.errors: list[BaseException] = []

    def now_ps(self) -> int:
        return (time.monotonic_ns() - self._t0) * 1000

    def call_later(self, delay_ps: int, callback: Callable[..., Any], *args: Any,
                   priority: int = 0) -> EventHandle:
        if delay_ps < 0:
            raise SimulationError("negative delay")
        with self._cv:
            h = EventHandle(self.now_ps() + int(delay_ps), priority, next(self._seq), callback, args)
            heapq.heappush(self._queue, h)
            self._cv.notify()
            return h

    def call_soon(self, callback: Callable[..., Any], *args: Any) -> EventHandle:
        return self.call_later(0, callback, *args)

    def cancel(self, handle: Optional[EventHandle]) -> None:
        if handle is not None:
            handle.cancelled = True

    def _loop(self) -> None:
        while True:
            with self._cv:
                while not self._closed:
                    while self._queue and self._queue[0].cancelled:
                        heapq.heappop(self._queue)
                    if not self._queue:
                        self._cv.wait()
                        continue
                    wait_ps = self._queue[0].time - self.now_ps()
                    if wait_ps <= 0:
                        break
                    self._cv.wait(timeout=wait_ps / 1e12)
                if self._closed:
                    return
                h = heapq.heappop(self._queue)
            try:
                h.callback(*h.args)
            except BaseException as exc:  # keep the dispatcher alive; surface later
                self.errors.append(exc)

    def close(self, timeout_s: float = 2.0) -> None:
        with self._cv:
            self._closed = True
            self._cv.notify_all()
        self._thread.join(timeout=timeout_s)

    def __enter__(self) -> "RealtimeScheduler":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

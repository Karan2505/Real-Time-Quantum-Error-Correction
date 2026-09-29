"""
AURA-Link time synchronisation (PTP-style two-step exchange over the link).

The control system is the time master (its clock timestamps every
measurement); the gateway disciplines its hardware clock to it so that
``arrival_ts - origin_ts`` is a true one-way latency.

  master                          slave
  TS_SYNC(seq)      ---t1--->     t2 = SOF timestamp
  TS_FOLLOW_UP(t1)  -------->
                    <---t3---     TS_DELAY_REQ(seq)
  TS_DELAY_RESP(t4) -------->     offset = ((t2-t1) - (t4-t3)) / 2
                                  delay  = ((t2-t1) + (t4-t3)) / 2

t1 and t3 are egress timestamps taken when the frame's first bit leaves the
wire (two-step), t2 and t4 are the TSU ingress timestamps.  The slave clock
is stepped once, then steered by a PI servo (same structure as linuxptp).
Path asymmetry appears as a fixed offset error of asymmetry/2 (inherent to
any two-way time transfer); ``asymmetry_ps`` can calibrate it out.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Callable, Optional

from ..core.clock import LocalClock
from ..core.telemetry import MetricsRegistry
from ..core.units import NS, US


@dataclass
class TimeSyncConfig:
    interval_ps: int = 20 * US              # sync exchange period
    step_threshold_ps: int = 1 * US         # step the clock if |offset| exceeds this
    kp: float = 0.7
    ki: float = 0.3
    lock_threshold_ps: int = 2 * NS         # |offset| below this ...
    lock_samples: int = 4                   # ... for this many consecutive samples => locked
    unlock_threshold_ps: int = 20 * NS
    asymmetry_ps: int = 0                   # known (calibrated) path asymmetry, slave->master minus master->slave
    max_ppm: float = 500.0


class PiServo:
    def __init__(self, cfg: TimeSyncConfig):
        self.cfg = cfg
        self.drift = 0.0          # integrated fractional frequency error
        self.first = True

    def sample(self, offset_ps: float, interval_ps: float) -> tuple[str, float]:
        """Return ('step', delta_ps) or ('freq', steer_fraction)."""
        c = self.cfg
        if self.first or abs(offset_ps) > c.step_threshold_ps:
            self.first = False
            return "step", -offset_ps
        ki_term = c.ki * offset_ps / interval_ps
        self.drift += ki_term
        f = -(c.kp * offset_ps / interval_ps + self.drift)
        lim = c.max_ppm * 1e-6
        return "freq", max(-lim, min(lim, f))


class TimeSyncMaster:
    def __init__(self, sched, clock: LocalClock, send: Callable, cfg: TimeSyncConfig, metrics: MetricsRegistry):
        self.s = sched
        self.clock = clock
        self.send = send                   # send(ftype, payload_bytes, on_wire_sof) -> None
        self.cfg = cfg
        self.m = metrics
        self._seq = 0
        self._timer = None
        self.running = False

    def start(self) -> None:
        if self.running:
            return
        self.running = True
        self._tick()

    def stop(self) -> None:
        self.running = False
        self.s.cancel(self._timer)
        self._timer = None

    def _tick(self) -> None:
        if not self.running:
            return
        from .frames import FrameType, encode_ts
        seq = self._seq
        self._seq = (self._seq + 1) & 0xFFFF

        def on_sof(_t: int, seq=seq) -> None:
            t1 = self.clock.timestamp()
            self.send(FrameType.TS_FOLLOW_UP, encode_ts(seq, t1), None)
        self.send(FrameType.TS_SYNC, encode_ts(seq, 0), on_sof)
        self.m.counter("timesync.sync_sent").inc()
        self._timer = self.s.call_later(self.cfg.interval_ps, self._tick)

    def on_delay_req(self, seq: int, t4: int) -> None:
        from .frames import FrameType, encode_ts
        self.send(FrameType.TS_DELAY_RESP, encode_ts(seq, t4), None)


class TimeSyncSlave:
    def __init__(self, sched, clock: LocalClock, send: Callable, cfg: TimeSyncConfig, metrics: MetricsRegistry,
                 on_lock_change: Optional[Callable[[bool], None]] = None):
        self.s = sched
        self.clock = clock
        self.send = send
        self.cfg = cfg
        self.m = metrics
        self.servo = PiServo(cfg)
        self._t2: dict[int, int] = {}
        self._t12: dict[int, tuple[int, int]] = {}
        self._t3: dict[int, int] = {}
        self._good = 0
        self.locked = False
        self.on_lock_change = on_lock_change
        self.last_offset_ps: Optional[float] = None
        self.last_delay_ps: Optional[float] = None
        self.history: deque = deque(maxlen=512)
        self._last_sample_t: Optional[int] = None

    def on_sync(self, seq: int, t2: int) -> None:
        self._t2[seq] = t2
        if len(self._t2) > 64:
            self._t2.pop(next(iter(self._t2)))

    def on_follow_up(self, seq: int, t1: int) -> None:
        t2 = self._t2.pop(seq, None)
        if t2 is None:
            self.m.counter("timesync.orphan_follow_up").inc()
            return
        self._t12[seq] = (t1, t2)
        from .frames import FrameType, encode_ts

        def on_sof(_t: int, seq=seq) -> None:
            self._t3[seq] = self.clock.timestamp()
        self.send(FrameType.TS_DELAY_REQ, encode_ts(seq, 0), on_sof)

    def on_delay_resp(self, seq: int, t4: int) -> None:
        t12 = self._t12.pop(seq, None)
        t3 = self._t3.pop(seq, None)
        if t12 is None or t3 is None:
            self.m.counter("timesync.incomplete_exchange").inc()
            return
        t1, t2 = t12
        ms = t2 - t1                       # master -> slave (includes offset)
        sm = t4 - t3                       # slave -> master (minus offset)
        offset = (ms - sm - self.cfg.asymmetry_ps) / 2.0
        delay = (ms + sm) / 2.0
        self.last_offset_ps = offset
        self.last_delay_ps = delay
        now = self.s.now_ps()
        interval = self.cfg.interval_ps if self._last_sample_t is None else max(1, now - self._last_sample_t)
        self._last_sample_t = now
        self.history.append((now, offset, delay))
        self.m.histogram("timesync.abs_offset_ps").record(int(abs(offset)))
        self.m.gauge("timesync.offset_ps").set(offset)
        self.m.gauge("timesync.path_delay_ps").set(delay)
        kind, v = self.servo.sample(offset, interval)
        if kind == "step":
            self.clock.step(int(round(v)))
            self.m.counter("timesync.steps").inc()
            self._good = 0
        else:
            self.clock.adjust_frequency(v * 1e6)
        # lock detection with hysteresis
        if abs(offset) <= self.cfg.lock_threshold_ps and kind == "freq":
            self._good += 1
        elif abs(offset) > self.cfg.unlock_threshold_ps:
            self._good = 0
        was = self.locked
        if not self.locked and self._good >= self.cfg.lock_samples:
            self.locked = True
        elif self.locked and abs(offset) > self.cfg.unlock_threshold_ps:
            self.locked = False
            self.m.counter("timesync.unlocks").inc()
        if was != self.locked and self.on_lock_change:
            self.on_lock_change(self.locked)

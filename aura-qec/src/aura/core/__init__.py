"""Core services: virtual-time kernel, clocks, RNG streams, telemetry, config."""

from .units import PS, NS, US, MS, S, ns, us, ms, to_ns, to_us, period_ps, ceil_to_grid, fmt_time
from .sim import Simulator, RealtimeScheduler, EventHandle, Scheduler, SimulationError
from .clock import LocalClock, TrueTime
from .rng import RngFactory
from .telemetry import MetricsRegistry, Histogram, Counter, Gauge
from .errors import (AuraError, ConfigError, CodeError, CompileError, TimingViolation,
                     HardwareFault, CalibrationError, LinkError, ProtocolError)

__all__ = [n for n in dir() if not n.startswith("_")]

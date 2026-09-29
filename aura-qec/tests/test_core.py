import pytest

from aura.core import (ConfigError, Histogram, LocalClock, RngFactory, SimulationError, Simulator,
                       TrueTime, us)
from aura.core.config import from_dict
from aura.link.endpoint import LinkConfig


def test_simulator_orders_by_time_priority_and_insertion():
    sim = Simulator()
    out = []
    sim.schedule(10, out.append, "b")
    sim.schedule(10, out.append, "a", priority=-1)
    sim.schedule(5, out.append, "first")
    sim.schedule(10, out.append, "c")
    sim.run()
    assert out == ["first", "a", "b", "c"]
    assert sim.now == 10


def test_simulator_cancel_until_and_reentrancy():
    sim = Simulator()
    out = []
    h = sim.schedule(5, out.append, 1)
    sim.schedule(20, out.append, 2)
    sim.cancel(h)
    sim.run(until_ps=10)
    assert out == [] and sim.now == 10
    with pytest.raises(SimulationError):
        sim.schedule(-1, out.append, 3)
    with pytest.raises(SimulationError):
        sim.schedule_at(5, out.append, 3)
    sim.schedule(0, lambda: sim.run())
    with pytest.raises(SimulationError):
        sim.run()


def test_simulator_rejects_float_times():
    with pytest.raises(TypeError):
        Simulator().schedule(1.5, lambda: None)


def test_local_clock_offset_rate_step_and_steering():
    sim = Simulator()
    clk = LocalClock(TrueTime(sim), offset_ps=1000, freq_ppm=100.0)
    sim.run(until_ps=us(10))
    assert clk.now_ps() == pytest.approx(us(10) + 1000 + us(10) * 1e-4, abs=2)
    clk.step(-1000)
    clk.adjust_frequency(-100.0)
    t = clk.now_ps()
    sim.run(until_ps=us(20))
    assert clk.now_ps() - t == pytest.approx(us(10), abs=2)


def test_rng_streams_are_order_independent():
    a = RngFactory(7)
    b = RngFactory(7)
    b.stream("other")
    assert a.stream("x").integers(1 << 30) == b.stream("x").integers(1 << 30)
    assert RngFactory(7).stream("x").integers(1 << 30) != RngFactory(8).stream("x").integers(1 << 30)


def test_histogram_percentiles_are_accurate():
    h = Histogram("h", 64)
    vals = list(range(1, 100001))
    h.record_many(vals)
    assert h.count == len(vals)
    assert h.percentile(50) == pytest.approx(50000, rel=0.02)
    assert h.percentile(99) == pytest.approx(99000, rel=0.02)
    assert h.percentile(100) == 100000 and h.percentile(0) == 1
    with pytest.raises(ValueError):
        h.record(-1)


def test_config_is_strict():
    with pytest.raises(ConfigError):
        from_dict(LinkConfig, {"soft_bitz": 4})
    with pytest.raises(ConfigError):
        from_dict(LinkConfig, {"soft_bits": "four"})
    with pytest.raises(Exception):   # 25G line needs a faster core clock for a 64-bit datapath
        from_dict(LinkConfig, {"phy": {"line_rate_gbps": 25.78125}})
    cfg = from_dict(LinkConfig, {"soft_bits": 8, "phy": {"line_rate_gbps": 25.78125, "core_clock_mhz": 400.0}})
    assert cfg.soft_bits == 8 and cfg.phy.line_rate_gbps == 25.78125

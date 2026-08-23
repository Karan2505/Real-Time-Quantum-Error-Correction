"""
conftest.py
===========
Pytest session-level configuration and shared fixtures.

Fixtures provided
-----------------
cfg         — parsed YAML config dict
qeci        — QECi interface (stub or real, per config)
stimulus    — StimulusGenerator with default params
ts_capture  — fresh TimestampCapture per test
collector   — fresh MetricsCollector per test
reporter    — ReportGenerator pointing at config output_dir
"""

from __future__ import annotations

import os
import sys

import pytest
import yaml

# Make project root importable regardless of working directory
sys.path.insert(0, os.path.dirname(__file__))

from harness.qeci_interface import create_interface, QECiInterface
from harness.stimulus_generator import StimulusGenerator
from harness.timestamp_capture import TimestampCapture
from metrics.collector import MetricsCollector
from reporting.report_generator import ReportGenerator

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config", "test_config.yaml")


# ---------------------------------------------------------------------------
# Session-scoped: config and reporter (created once per pytest run)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def cfg() -> dict:
    """Parse and return the YAML test configuration."""
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="session")
def reporter(cfg) -> ReportGenerator:
    """Shared report generator pointing at config.reporting.output_dir."""
    out_dir = cfg.get("reporting", {}).get("output_dir", "reports")
    keep = cfg.get("reporting", {}).get("keep_raw_traces", True)
    return ReportGenerator(output_dir=out_dir, keep_raw_traces=keep)


# ---------------------------------------------------------------------------
# Function-scoped: fresh for every test
# ---------------------------------------------------------------------------

@pytest.fixture
def qeci(cfg) -> QECiInterface:
    """
    Return the appropriate QECi interface (stub or real HW).
    Closed automatically after each test.
    """
    iface = create_interface(cfg)
    yield iface
    iface.close()


@pytest.fixture
def stimulus(cfg) -> StimulusGenerator:
    """Stimulus generator with parameters from config."""
    return StimulusGenerator(
        num_qubits=cfg.get("stimulus", {}).get("measurement_bits", 8),
        error_rate=0.001,
        seed=42,
    )


@pytest.fixture
def ts_capture(cfg) -> TimestampCapture:
    """Fresh TimestampCapture per test."""
    skew_tol = cfg.get("timing", {}).get("max_clock_skew_us", 50.0)
    return TimestampCapture(max_clock_skew_us=skew_tol)


@pytest.fixture
def collector() -> MetricsCollector:
    """Fresh MetricsCollector per test."""
    return MetricsCollector()


# ---------------------------------------------------------------------------
# Helpers available to all tests
# ---------------------------------------------------------------------------

def run_chunk_and_record(
    chunk_id: int,
    chunk_size: int,
    qeci: QECiInterface,
    stimulus: StimulusGenerator,
    ts_capture: TimestampCapture,
    collector: MetricsCollector,
    timeout_s: float = 2.0,
) -> bool:
    """
    Send one chunk, wait for the correction result, and record metrics.

    Returns True if a valid result was received.
    """
    from harness.qeci_interface import Chunk
    import time

    chunk = stimulus.make_chunk(chunk_id=chunk_id, size=chunk_size)
    ts_capture.record_ingress(chunk_id=chunk_id,
                              timestamp_ns=chunk.final_measurement_time_ns)
    qeci.send_chunk(chunk)

    result = qeci.receive_correction(timeout_s=timeout_s)
    if result is None:
        collector.record_timeout(chunk_id=chunk_id, n_measurements=chunk_size)
        return False

    ts_capture.record_egress(chunk_id=result.chunk_id,
                             timestamp_ns=result.egress_time_ns)
    latency_ns = ts_capture.compute_latency_ns(result.chunk_id) or 0.0
    valid = len(result.correction_ops) > 0

    collector.record_chunk_result(
        chunk_id=result.chunk_id,
        n_measurements=chunk_size,
        latency_ns=latency_ns,
        result_valid=valid,
    )
    return valid

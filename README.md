# Deltaflow System Integration and Validation Test Framework

Python skeleton test framework for validating and verifying Riverlane's Deltaflow
QEC stack in a lab environment without physical qubits or a live control system.

## Directory Structure

```
deltaflow_test/
├── README.md
├── requirements.txt
├── conftest.py                        # Pytest fixtures and session setup
├── config/
│   └── test_config.yaml               # All tuneable parameters in one place
├── harness/
│   ├── qeci_interface.py              # QECi transport abstraction (real or stub)
│   ├── stimulus_generator.py          # Deterministic measurement stream generator
│   ├── timestamp_capture.py           # Ingress/egress timestamping with sync
│   └── impairment_emulator.py         # Delay, jitter, loss, reorder injection
├── fixtures/
│   └── syndrome_traces.py             # Known syndrome traces for replay/oracle
├── tests/
│   ├── test_smoke.py                  # Step 1 — QECi connectivity + wiring
│   ├── test_chunking.py               # Step 2 — Batching boundary correctness
│   ├── test_decoder_replay.py         # Step 3 — Decoder correctness vs oracle
│   ├── test_decoder_robustness.py     # Step 4 — Decoder under hostile traces
│   ├── test_throughput.py             # Step 5a — Throughput sweep (meas/s)
│   ├── test_latency.py                # Step 5b — Chunk-level latency distribution
│   └── test_burst_recovery.py         # Step 6 — Burst, overload, recovery
├── metrics/
│   └── collector.py                   # Stats: mean/p95/p99/max, throughput calc
└── reporting/
    └── report_generator.py            # JSON + human-readable run reports
```

## Execution Order

Run tests in the order recommended in the test plan:

```
1  Smoke            →  pytest tests/test_smoke.py
2  Chunking         →  pytest tests/test_chunking.py
3  Decoder replay   →  pytest tests/test_decoder_replay.py
4  Decoder stress   →  pytest tests/test_decoder_robustness.py
5a Throughput       →  pytest tests/test_throughput.py
5b Latency          →  pytest tests/test_latency.py
6  Burst/recovery   →  pytest tests/test_burst_recovery.py

Full suite         →  pytest tests/ -v --tb=short
```

## Key Performance Formulae

```
Throughput  T  = N_processed  /  Δt          (measurements per second)
Chunk latency  L_chunk = t_egress − t_final_input
Decoder rate   R_decode = C_decoded / Δt     (chunks per second)
Oracle agreement  A = M_match / N_total
```

## Assumptions

- No physical qubits or live control system available.
- QECi transport is either the real hardware endpoint or the software stub
  (`harness/qeci_interface.py`) depending on `USE_REAL_HW` in config.
- Clock synchronization (PTP or shared reference) is assumed available for
  sub-millisecond latency accuracy; see `harness/timestamp_capture.py`.
- Syndrome traces in `fixtures/syndrome_traces.py` are deterministic stand-ins
  until real qubit-derived captures are available.

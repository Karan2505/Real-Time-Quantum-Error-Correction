"""
reporting/report_generator.py
==============================
Test run report generator.

Produces two outputs per test run:
  1. A JSON file with machine-readable metrics (for CI comparison).
  2. A human-readable text summary (for engineers reviewing results).

Structure matches the test plan section 11 (Reporting and deliverables):
  * Executive summary (pass/fail, headline metrics)
  * Tabulated metrics
  * Notes section (anomalies, warnings)
  * Raw trace path reference

Usage
-----
from reporting.report_generator import ReportGenerator

rg = ReportGenerator(output_dir="reports")
rg.write(
    test_name="test_throughput_sweep",
    config=cfg,
    metrics=collector.report_dict(),
    passed=True,
    notes=["Rate 100k mps: backlog growing — knee point identified"],
)
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


class ReportGenerator:
    """
    Writes JSON and plain-text reports for each test run.

    Parameters
    ----------
    output_dir : str
        Directory where reports are written.  Created if absent.
    keep_raw_traces : bool
        If True, a placeholder raw-trace file path is included in the report.
    """

    def __init__(self, output_dir: str = "reports", keep_raw_traces: bool = True):
        self.output_dir = output_dir
        self.keep_raw_traces = keep_raw_traces
        os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def write(
        self,
        test_name: str,
        config: Dict[str, Any],
        metrics: Dict[str, Any],
        passed: bool,
        notes: Optional[List[str]] = None,
        raw_trace_path: Optional[str] = None,
    ) -> str:
        """
        Write both JSON and text reports for a test run.

        Returns the path to the JSON report file.
        """
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_id = f"{test_name}__{timestamp}"

        report = self._build_report(
            run_id=run_id,
            test_name=test_name,
            config=config,
            metrics=metrics,
            passed=passed,
            notes=notes or [],
            raw_trace_path=raw_trace_path,
        )

        json_path = os.path.join(self.output_dir, f"{run_id}.json")
        txt_path  = os.path.join(self.output_dir, f"{run_id}.txt")

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(self._format_text(report))

        return json_path

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_report(
        self,
        run_id: str,
        test_name: str,
        config: dict,
        metrics: dict,
        passed: bool,
        notes: List[str],
        raw_trace_path: Optional[str],
    ) -> dict:
        report: Dict[str, Any] = {
            "run_id": run_id,
            "test_name": test_name,
            "status": "PASS" if passed else "FAIL",
            "utc_timestamp": datetime.now(timezone.utc).isoformat() + "Z",
            "config_snapshot": {
                "chunk_sizes": config.get("stimulus", {}).get("chunk_sizes"),
                "rates_mps": config.get("stimulus", {}).get("rates_mps"),
                "use_real_hw": config.get("deltaflow", {}).get("use_real_hw"),
                "response_timeout_s": config.get("deltaflow", {}).get("response_timeout_s"),
            },
            "metrics": metrics,
            "acceptance": {
                "min_throughput_mps": config.get("performance", {}).get("min_throughput_mps"),
                "max_p99_latency_ms": config.get("performance", {}).get("max_p99_latency_ms"),
                "max_mean_latency_ms": config.get("performance", {}).get("max_mean_latency_ms"),
            },
            "notes": notes,
        }
        if self.keep_raw_traces:
            report["raw_trace_path"] = raw_trace_path or f"traces/{run_id}.bin"
        return report

    def _format_text(self, report: dict) -> str:
        lines = [
            "=" * 72,
            f"  Deltaflow Test Run Report",
            f"  Run ID   : {report['run_id']}",
            f"  Status   : {report['status']}",
            f"  Timestamp: {report['utc_timestamp']}",
            "=" * 72,
            "",
            "[ Executive Summary ]",
            f"  Test         : {report['test_name']}",
            f"  Result       : {'✓ PASS' if report['status'] == 'PASS' else '✗ FAIL'}",
            "",
        ]

        m   = report.get("metrics", {})
        cmp = m.get("comparison", {})   # present in burst/recovery tests only
        before_m = cmp.get("before", {})

        # ---- Pre/Post Comparison block (burst/recovery tests only) ----------
        if cmp:
            lines += ["[ Pre/Post Comparison ]"]
            lines.append(
                f"  Before throughput : "
                f"{cmp.get('tput_before_mps', before_m.get('throughput_mps', 'N/A'))} meas/s"
            )
            lines.append(
                f"  After throughput  : "
                f"{cmp.get('tput_after_mps', m.get('throughput_mps', 'N/A'))} meas/s"
            )
            if "recovery_ratio" in cmp:
                r = cmp["recovery_ratio"]
                lines.append(f"  Recovery ratio    : {r:.3f}  ({r * 100:.1f}%)")
            if "drift_pct" in cmp:
                lines.append(f"  Throughput drift  : {cmp['drift_pct']}%")
            if "burst_multiplier" in cmp:
                lines.append(f"  Burst multiplier  : {cmp['burst_multiplier']}x")
            if "loss_probability" in cmp:
                lines.append(f"  Loss probability  : {cmp['loss_probability']:.0%}")
            if "lossy_window_sent" in cmp:
                lines.append(
                    f"  Lossy window      : {cmp['lossy_window_sent']} chunks sent, "
                    f"{cmp.get('lossy_window_dropped', 'N/A')} dropped"
                )
            blat = before_m.get("latency", {})
            if blat and blat.get("count", 0) > 0:
                lines += [
                    "",
                    "  -- Before phase latency --",
                    f"  Mean / p99 / Max  : "
                    f"{blat.get('mean_ms', 'N/A')} / "
                    f"{blat.get('p99_ms', 'N/A')} / "
                    f"{blat.get('max_ms', 'N/A')} ms",
                ]
            lines.append("")

        # ---- Throughput block -----------------------------------------------
        header = "[ Throughput (After Phase) ]" if cmp else "[ Throughput ]"
        label  = "After throughput " if cmp else "Processed meas/s"
        lines += [
            header,
            f"  {label} : {m.get('throughput_mps', 'N/A')} meas/s",
            f"  Decoder chunks/s : {m.get('decoder_rate_cps', 'N/A')}",
            f"  Observation time : {m.get('elapsed_s', 'N/A')} s",
            f"  Total chunks     : {m.get('total_chunks', 'N/A')}",
            f"  Valid / Invalid  : {m.get('valid_chunks', 'N/A')} / {m.get('invalid_chunks', 'N/A')}",
            "",
        ]

        # ---- Latency block --------------------------------------------------
        lat = m.get("latency", {})
        if lat and lat.get("count", 0) > 0:
            phase = "After Phase " if cmp else ""
            lines += [
                f"[ {phase}Latency Distribution ]",
                f"  Count   : {lat.get('count', 'N/A')} chunks",
                f"  Mean    : {lat.get('mean_ms', 'N/A')} ms",
                f"  Median  : {lat.get('median_ms', 'N/A')} ms",
                f"  p95     : {lat.get('p95_ms', 'N/A')} ms",
                f"  p99     : {lat.get('p99_ms', 'N/A')} ms",
                f"  Max     : {lat.get('max_ms', 'N/A')} ms",
                f"  Std Dev : {lat.get('std_ms', 'N/A')} ms",
                "",
            ]

        acc = report.get("acceptance", {})
        lines += [
            "[ Acceptance Criteria ]",
            f"  Min throughput  : {acc.get('min_throughput_mps', 'N/A')} meas/s",
            f"  Max p99 latency : {acc.get('max_p99_latency_ms', 'N/A')} ms",
            f"  Max mean latency: {acc.get('max_mean_latency_ms', 'N/A')} ms",
            "",
        ]

        notes = report.get("notes", [])
        if notes:
            lines.append("[ Notes ]")
            for n in notes:
                lines.append(f"  * {n}")
            lines.append("")

        if report.get("raw_trace_path"):
            lines += [
                "[ Raw Trace ]",
                f"  Path: {report['raw_trace_path']}",
                "",
            ]

        lines.append("=" * 72)
        return "\n".join(lines)
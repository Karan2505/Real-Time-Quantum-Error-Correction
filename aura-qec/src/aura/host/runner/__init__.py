"""Experiment runner results and statistics (orchestration lives in aura.lab.VirtualLab)."""

from .results import RunResult, PatchResult, wilson_interval, latency_stats

__all__ = ["RunResult", "PatchResult", "wilson_interval", "latency_stats"]

"""
rl
==
RL controller research package for AURA-QEC.

This package is developed independently of Karan's Deltaflow validation
framework (harness/, metrics/, reporting/, etc.) and does not import from
or modify it. It is intentionally decoupled behind ``DecoderProtocol``
(see rl/environment.py) so the mock decoder used in Phase 1 can later be
swapped for a real Relay-BP implementation without touching the
environment, reward, or (eventually) RL-agent code.

STATUS (Phase 1): infrastructure only. The decoder backing this package
is a synthetic MOCK (see rl/mock_decoder.py) and no output produced by
this package should be reported as a QEC research result.
"""

from __future__ import annotations

__all__ = [
    "actions",
    "config",
    "state",
    "mock_decoder",
    "environment",
    "reward",
]

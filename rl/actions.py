"""
rl/actions.py
=============
Fixed action space for the RL decoding controller.

This mapping is part of the RL policy's public interface: a trained
Q-network's output layer will have one unit per action, in this exact
order. Do not reorder these values once training begins, or saved
policies/checkpoints will silently become invalid.
"""

from __future__ import annotations

from enum import IntEnum


class Action(IntEnum):
    """Decoding actions the RL controller may select.

    CONTINUE — run another Relay-BP iteration (or, in Phase 1, advance
               the mock decoder by one synthetic iteration).
    STOP     — terminate decoding and accept the current correction.
    OSD      — invoke OSD-lite (Phase 1: a labeled mock fallback) as a
               fallback for a stagnating/difficult instance.
    """

    CONTINUE = 0
    STOP = 1
    OSD = 2


#: Convenience set used for fast membership / validity checks.
VALID_ACTIONS = frozenset(int(a) for a in Action)


def coerce_action(value: int | Action) -> Action:
    """Validate and coerce ``value`` into an :class:`Action`.

    Raises
    ------
    ValueError
        If ``value`` is not one of the three defined actions.
    """
    if isinstance(value, bool) or not isinstance(value, (int, Action)):
        raise ValueError(
            f"Invalid action {value!r}; expected an int or Action, one of "
            f"{sorted(VALID_ACTIONS)} (CONTINUE=0, STOP=1, OSD=2)."
        )
    try:
        return Action(int(value))
    except ValueError as exc:
        raise ValueError(
            f"Invalid action {value!r}; expected one of "
            f"{sorted(VALID_ACTIONS)} (CONTINUE=0, STOP=1, OSD=2)."
        ) from exc

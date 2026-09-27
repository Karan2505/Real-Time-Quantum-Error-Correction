"""Unit tests for rl/actions.py."""

from __future__ import annotations

import pytest

from rl.actions import Action, VALID_ACTIONS, coerce_action


class TestActionEnum:
    def test_fixed_values(self):
        # These specific integer values are part of the RL policy's
        # public interface (DQN output-layer ordering) and must never
        # silently change.
        assert Action.CONTINUE == 0
        assert Action.STOP == 1
        assert Action.OSD == 2

    def test_three_actions_defined(self):
        assert len(Action) == 3

    def test_valid_actions_set(self):
        assert VALID_ACTIONS == {0, 1, 2}


class TestCoerceAction:
    @pytest.mark.parametrize("value,expected", [(0, Action.CONTINUE), (1, Action.STOP), (2, Action.OSD)])
    def test_int_coercion(self, value, expected):
        assert coerce_action(value) is expected

    @pytest.mark.parametrize("value,expected", [(Action.CONTINUE, Action.CONTINUE), (Action.OSD, Action.OSD)])
    def test_action_passthrough(self, value, expected):
        assert coerce_action(value) is expected

    @pytest.mark.parametrize("bad_value", [3, -1, 100])
    def test_out_of_range_raises(self, bad_value):
        with pytest.raises(ValueError):
            coerce_action(bad_value)

    @pytest.mark.parametrize("bad_value", ["CONTINUE", None, 1.5, object()])
    def test_wrong_type_raises(self, bad_value):
        with pytest.raises(ValueError):
            coerce_action(bad_value)

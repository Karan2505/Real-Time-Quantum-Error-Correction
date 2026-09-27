"""Unit tests for rl/state.py."""

from __future__ import annotations

import numpy as np
import pytest

from rl.state import DecoderState, StateMode


def make_state(**overrides) -> DecoderState:
    defaults = dict(
        unsatisfied_check_ratio=0.3,
        delta_LLR=0.05,
        current_iteration=4,
        physical_error_rate=0.01,
        iterations_since_improvement=2,
    )
    defaults.update(overrides)
    return DecoderState(**defaults)


class TestConstruction:
    def test_valid_construction(self):
        s = make_state()
        assert s.unsatisfied_check_ratio == 0.3
        assert s.current_iteration == 4
        assert s.iterations_since_improvement == 2

    @pytest.mark.parametrize(
        "field,value",
        [
            ("unsatisfied_check_ratio", -0.01),
            ("unsatisfied_check_ratio", 1.01),
            ("physical_error_rate", -0.5),
            ("physical_error_rate", 1.5),
        ],
    )
    def test_out_of_range_ratio_rejected(self, field, value):
        with pytest.raises(ValueError):
            make_state(**{field: value})

    def test_negative_current_iteration_rejected(self):
        with pytest.raises(ValueError):
            make_state(current_iteration=-1)

    def test_negative_iterations_since_improvement_rejected(self):
        with pytest.raises(ValueError):
            make_state(iterations_since_improvement=-1)

    def test_boundary_values_accepted(self):
        # 0 and 1 are valid boundary values for ratio-like fields.
        s = make_state(unsatisfied_check_ratio=0.0, physical_error_rate=1.0)
        assert s.unsatisfied_check_ratio == 0.0
        assert s.physical_error_rate == 1.0


class TestVectorConversion:
    def test_oracle_vector_shape_and_dtype(self):
        s = make_state()
        vec = s.to_vector(StateMode.ORACLE)
        assert vec.shape == (5,)
        assert vec.dtype == np.float32

    def test_realistic_vector_shape(self):
        s = make_state()
        vec = s.to_vector(StateMode.REALISTIC)
        assert vec.shape == (4,)

    def test_oracle_feature_ordering(self):
        s = make_state(
            unsatisfied_check_ratio=0.11,
            delta_LLR=0.22,
            current_iteration=3,
            physical_error_rate=0.44,
            iterations_since_improvement=5,
        )
        vec = s.to_vector(StateMode.ORACLE)
        expected = np.array([0.11, 0.22, 3.0, 0.44, 5.0], dtype=np.float32)
        np.testing.assert_allclose(vec, expected, rtol=1e-6)

    def test_realistic_feature_ordering_excludes_physical_error_rate(self):
        s = make_state(
            unsatisfied_check_ratio=0.11,
            delta_LLR=0.22,
            current_iteration=3,
            physical_error_rate=0.999,  # should not appear below
            iterations_since_improvement=5,
        )
        vec = s.to_vector(StateMode.REALISTIC)
        expected = np.array([0.11, 0.22, 3.0, 5.0], dtype=np.float32)
        np.testing.assert_allclose(vec, expected, rtol=1e-6)
        assert 0.999 not in vec.tolist()

    def test_default_mode_is_oracle(self):
        s = make_state()
        default_vec = s.to_vector()
        oracle_vec = s.to_vector(StateMode.ORACLE)
        np.testing.assert_array_equal(default_vec, oracle_vec)

    def test_vector_size_matches_feature_names(self):
        assert DecoderState.vector_size(StateMode.ORACLE) == len(
            DecoderState.feature_names(StateMode.ORACLE)
        )
        assert DecoderState.vector_size(StateMode.REALISTIC) == len(
            DecoderState.feature_names(StateMode.REALISTIC)
        )
        assert DecoderState.vector_size(StateMode.ORACLE) == 5
        assert DecoderState.vector_size(StateMode.REALISTIC) == 4

    def test_feature_names_first_three_entries_match_across_modes(self):
        # Both modes should agree on the shared, non-oracle-only prefix
        # of features (order must not silently diverge between modes).
        oracle_names = DecoderState.feature_names(StateMode.ORACLE)
        realistic_names = DecoderState.feature_names(StateMode.REALISTIC)
        assert oracle_names[0] == realistic_names[0] == "unsatisfied_check_ratio"
        assert oracle_names[1] == realistic_names[1] == "delta_LLR"
        assert oracle_names[2] == realistic_names[2] == "current_iteration"
        assert "physical_error_rate" in oracle_names
        assert "physical_error_rate" not in realistic_names

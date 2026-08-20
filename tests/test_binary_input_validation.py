import numpy as np
import pytest
from polars.testing import assert_frame_equal

from rtichoke import prepare_performance_data


def test_binary_reals_dict_is_aligned_by_group_key_not_insertion_order():
    probs = {
        "group_a": np.array([0.1, 0.8, 0.7, 0.2]),
        "group_b": np.array([0.9, 0.2, 0.3, 0.8]),
    }
    reals_a = np.array([0, 1, 1, 0])
    reals_b = np.array([1, 0, 0, 1])

    aligned = prepare_performance_data(
        probs=probs,
        reals={"group_a": reals_a, "group_b": reals_b},
        by=0.5,
    )
    reversed_order = prepare_performance_data(
        probs=probs,
        reals={"group_b": reals_b, "group_a": reals_a},
        by=0.5,
    )

    assert_frame_equal(aligned, reversed_order)


def test_binary_single_group_reals_dict_matches_array_input():
    probs = {"group_a": np.array([0.1, 0.8, 0.7, 0.2])}
    reals = np.array([0, 1, 1, 0])

    from_array = prepare_performance_data(probs=probs, reals=reals, by=0.5)
    from_dict = prepare_performance_data(
        probs=probs,
        reals={"group_a": reals},
        by=0.5,
    )

    assert_frame_equal(from_array, from_dict)


def test_binary_reals_dict_keys_must_match_probability_groups():
    probs = {
        "group_a": np.array([0.1, 0.8]),
        "group_b": np.array([0.9, 0.2]),
    }

    with pytest.raises(ValueError, match="keys must exactly match"):
        prepare_performance_data(
            probs=probs,
            reals={
                "group_a": np.array([0, 1]),
                "wrong_group": np.array([1, 0]),
            },
            by=0.5,
        )


def test_binary_input_lengths_must_match_within_group():
    probs = {"group_a": np.array([0.1, 0.8, 0.7])}

    with pytest.raises(ValueError, match="Input lengths must match"):
        prepare_performance_data(
            probs=probs,
            reals={"group_a": np.array([0, 1])},
            by=0.5,
        )

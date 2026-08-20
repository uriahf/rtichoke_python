import numpy as np
import pytest

from rtichoke import prepare_performance_data_times


PROBS = {
    "train": np.array([0.1, 0.4, 0.7, 0.9]),
    "test": np.array([0.2, 0.3, 0.6, 0.8]),
}
REALS = {
    "train": np.array([0, 1, 0, 1]),
    "test": np.array([0, 0, 1, 1]),
}
TIMES = {
    "train": np.array([8.0, 2.0, 7.0, 3.0]),
    "test": np.array([9.0, 8.0, 4.0, 2.0]),
}


def _call(probs=PROBS, reals=REALS, times=TIMES):
    return prepare_performance_data_times(
        probs=probs,
        reals=reals,
        times=times,
        fixed_time_horizons=[5.0],
        by=0.5,
    )


def test_multiple_population_keys_must_match_probs():
    bad_reals = {"train": REALS["train"], "validation": REALS["test"]}

    with pytest.raises(ValueError, match="keys must exactly match `probs`"):
        _call(reals=bad_reals)


def test_multiple_groups_reject_mixed_dict_and_array_outcomes():
    with pytest.raises(ValueError, match="must both be arrays or both be dictionaries"):
        _call(reals=REALS, times=TIMES["train"])


def test_multiple_population_lengths_must_match_within_group():
    bad_times = {**TIMES, "test": TIMES["test"][:-1]}

    with pytest.raises(ValueError, match="Input lengths must match within group 'test'"):
        _call(times=bad_times)


def test_multiple_models_can_share_array_outcomes():
    shared_reals = np.array([0, 1, 0, 1])
    shared_times = np.array([8.0, 2.0, 7.0, 3.0])

    result = _call(reals=shared_reals, times=shared_times)

    assert set(result["reference_group"].to_list()) == {"train", "test"}


def test_single_group_accepts_keyed_reals_with_array_times():
    probs = {"train": PROBS["train"]}
    reals = {"train": REALS["train"]}

    result = _call(probs=probs, reals=reals, times=TIMES["train"])

    assert result.height > 0

import numpy as np
import pytest

from rtichoke.calibration._weights import _prepare_calibration_bins
from rtichoke.calibration.calibration import _make_deciles_dat_binary


def test_unweighted_private_calibration_bins_preserve_factual_semantics():
    probs = np.linspace(0.05, 0.95, 20)
    reals = np.tile(np.array([0, 1]), 10)

    expected = _make_deciles_dat_binary({"model": probs}, reals)
    actual = _prepare_calibration_bins(probs, reals)

    assert actual.select(expected.columns).equals(expected)
    assert actual.get_column("outcome_weight_sum").to_list() == [
        float(value) for value in expected.get_column("n").to_list()
    ]


def test_outcome_weights_change_observed_calibration_not_bins_or_predicted_means():
    probs = np.array([0.10, 0.20, 0.30, 0.40, 0.60, 0.70, 0.80, 0.90])
    reals = np.array([0, 1, 0, 1, 1, 0, 1, 0])
    weights = np.array([1, 3, 1, 1, 2, 1, 4, 1], dtype=float)

    weighted = _prepare_calibration_bins(
        probs, reals, outcome_weights=weights, n_bins=2
    )
    unweighted = _make_deciles_dat_binary({"model": probs}, reals, n_bins=2)

    assert weighted.get_column("decile").to_list() == unweighted.get_column(
        "decile"
    ).to_list()
    np.testing.assert_allclose(
        weighted.get_column("x").to_numpy(), unweighted.get_column("x").to_numpy()
    )
    np.testing.assert_allclose(weighted.get_column("y").to_numpy(), [4 / 6, 6 / 8])
    np.testing.assert_allclose(
        weighted.get_column("outcome_weight_sum").to_numpy(), [6, 8]
    )
    np.testing.assert_allclose(
        weighted.get_column("weighted_sum_reals").to_numpy(), [4, 6]
    )


def test_all_one_outcome_weights_reproduce_unweighted_bin_estimates():
    probs = np.linspace(0.05, 0.95, 20)
    reals = np.tile(np.array([0, 1]), 10)

    unweighted = _prepare_calibration_bins(probs, reals)
    weighted = _prepare_calibration_bins(
        probs, reals, outcome_weights=np.ones(reals.shape[0])
    )

    np.testing.assert_allclose(
        weighted.get_column("x").to_numpy(), unweighted.get_column("x").to_numpy()
    )
    np.testing.assert_allclose(
        weighted.get_column("y").to_numpy(), unweighted.get_column("y").to_numpy()
    )
    assert weighted.get_column("n_reals").to_list() == unweighted.get_column(
        "n_reals"
    ).to_list()
    assert weighted.get_column("n").to_list() == unweighted.get_column("n").to_list()


@pytest.mark.parametrize(
    "weights, message",
    [
        (np.array([1.0, 1.0]), "same length"),
        (np.array([1.0, -1.0, 1.0, 1.0]), "finite, non-negative"),
        (np.array([1.0, np.inf, 1.0, 1.0]), "finite, non-negative"),
    ],
)
def test_private_weighted_calibration_bins_validate_weights(weights, message):
    probs = np.array([0.1, 0.2, 0.8, 0.9])
    reals = np.array([0, 1, 1, 0])

    with pytest.raises(ValueError, match=message):
        _prepare_calibration_bins(probs, reals, outcome_weights=weights, n_bins=2)


def test_private_weighted_calibration_bins_require_positive_weight_in_each_bin():
    probs = np.array([0.1, 0.2, 0.8, 0.9])
    reals = np.array([0, 1, 1, 0])

    with pytest.raises(ValueError, match="positive total"):
        _prepare_calibration_bins(
            probs,
            reals,
            outcome_weights=np.array([1.0, 1.0, 0.0, 0.0]),
            n_bins=2,
        )


def test_private_weighted_calibration_bins_validate_bin_count():
    probs = np.array([0.1, 0.2])
    reals = np.array([0, 1])

    with pytest.raises(ValueError, match="positive integer"):
        _prepare_calibration_bins(probs, reals, n_bins=0)

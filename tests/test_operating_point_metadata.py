import numpy as np
import pytest

from rtichoke._viz_spec_v2 import (
    _roc_v2_spec_from_performance_data,
    _precision_recall_v2_spec_from_performance_data,
    _gains_v2_spec_from_performance_data,
    _lift_v2_spec_from_performance_data,
    _precision_recall_times_v2_spec_from_performance_data,
    _gains_times_v2_spec_from_performance_data,
    _lift_times_v2_spec_from_performance_data,
)
from rtichoke._decision_curve_viz_spec_v2 import (
    _decision_curve_v2_spec_from_performance_data,
    _decision_curve_times_v2_spec_from_performance_data,
)
from rtichoke._interventions_avoided_viz_spec_v2 import (
    _interventions_avoided_v2_spec_from_performance_data,
    _interventions_avoided_times_v2_spec_from_performance_data,
)
from rtichoke.performance_data.performance_data import prepare_performance_data
from rtichoke.performance_data.performance_data_times import (
    prepare_performance_data_times,
)
from rtichoke.processing.evaluation_semantics import _build_evaluation_metadata


@pytest.fixture
def sample_static_data():
    probs = {"Model A": np.array([0.1, 0.4, 0.7, 0.9])}
    reals = {"Model A": np.array([0, 0, 1, 1], dtype=np.float64)}
    perf_data = prepare_performance_data(probs, reals)
    metadata = _build_evaluation_metadata(probs, reals, np.array([]))
    return perf_data, metadata


@pytest.fixture
def sample_times_data():
    probs = {"Model A": np.array([0.1, 0.4, 0.7, 0.9, 0.2, 0.3, 0.8, 0.95])}
    reals = {"Model A": np.array([0, 0, 1, 1, 0, 1, 0, 1], dtype=np.float64)}
    times = {"Model A": np.array([1.0, 5.0, 3.0, 4.0, 2.0, 6.0, 7.0, 8.0])}
    perf_data = prepare_performance_data_times(
        probs, reals, times, fixed_time_horizons=[5.0]
    )
    metadata = _build_evaluation_metadata(probs, reals, times)
    return perf_data, metadata


def test_static_canonical_builders_operating_point_dimension(sample_static_data):
    perf_data, metadata = sample_static_data

    # ROC
    spec_roc_thresh = _roc_v2_spec_from_performance_data(
        perf_data, metadata, operating_point_dimension="probability_threshold"
    )
    assert spec_roc_thresh["operatingPoint"]["dimension"] == "probability_threshold"

    spec_roc_ppcr = _roc_v2_spec_from_performance_data(
        perf_data, metadata, operating_point_dimension="ppcr"
    )
    assert spec_roc_ppcr["operatingPoint"]["dimension"] == "ppcr"

    # PR
    spec_pr_thresh = _precision_recall_v2_spec_from_performance_data(
        perf_data, metadata, operating_point_dimension="probability_threshold"
    )
    assert spec_pr_thresh["operatingPoint"]["dimension"] == "probability_threshold"

    spec_pr_ppcr = _precision_recall_v2_spec_from_performance_data(
        perf_data, metadata, operating_point_dimension="ppcr"
    )
    assert spec_pr_ppcr["operatingPoint"]["dimension"] == "ppcr"

    # Gains
    spec_gains_thresh = _gains_v2_spec_from_performance_data(
        perf_data, metadata, operating_point_dimension="probability_threshold"
    )
    assert spec_gains_thresh["operatingPoint"]["dimension"] == "probability_threshold"

    # Lift
    spec_lift_ppcr = _lift_v2_spec_from_performance_data(
        perf_data, metadata, operating_point_dimension="ppcr"
    )
    assert spec_lift_ppcr["operatingPoint"]["dimension"] == "ppcr"

    # Utility
    spec_dc = _decision_curve_v2_spec_from_performance_data(
        perf_data, metadata, operating_point_dimension="probability_threshold"
    )
    assert spec_dc["operatingPoint"]["dimension"] == "probability_threshold"

    spec_ia = _interventions_avoided_v2_spec_from_performance_data(
        perf_data, metadata, operating_point_dimension="probability_threshold"
    )
    assert spec_ia["operatingPoint"]["dimension"] == "probability_threshold"


def test_time_canonical_builders_operating_point_dimension(sample_times_data):
    perf_data, metadata = sample_times_data

    spec_pr_t = _precision_recall_times_v2_spec_from_performance_data(
        perf_data, metadata, operating_point_dimension="probability_threshold"
    )
    assert spec_pr_t["operatingPoint"]["dimension"] == "probability_threshold"
    assert "horizon" in spec_pr_t["series"][0]

    spec_gains_t = _gains_times_v2_spec_from_performance_data(
        perf_data, metadata, operating_point_dimension="ppcr"
    )
    assert spec_gains_t["operatingPoint"]["dimension"] == "ppcr"
    assert "horizon" in spec_gains_t["series"][0]

    spec_lift_t = _lift_times_v2_spec_from_performance_data(
        perf_data, metadata, operating_point_dimension="probability_threshold"
    )
    assert spec_lift_t["operatingPoint"]["dimension"] == "probability_threshold"
    assert "horizon" in spec_lift_t["series"][0]

    spec_dc_t = _decision_curve_times_v2_spec_from_performance_data(
        perf_data, metadata, operating_point_dimension="probability_threshold"
    )
    assert spec_dc_t["operatingPoint"]["dimension"] == "probability_threshold"
    assert "horizon" in spec_dc_t["series"][0]

    spec_ia_t = _interventions_avoided_times_v2_spec_from_performance_data(
        perf_data, metadata, operating_point_dimension="probability_threshold"
    )
    assert spec_ia_t["operatingPoint"]["dimension"] == "probability_threshold"
    assert "horizon" in spec_ia_t["series"][0]

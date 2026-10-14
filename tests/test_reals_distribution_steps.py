import polars as pl
import pytest

from rtichoke import create_reals_distribution_steps
from rtichoke.reals_distribution import _outcome_distribution_steps_v2_spec


def test_shared_fixture_independent_numerical_reference():
    data = pl.DataFrame(
        {
            "subject": ["A", "B", "C", "A", "B", "C", "A", "C"],
            "step": [0, 0, 0, 1, 1, 1, 3, 3],
            "state": ["Low", "Moderate", "High", "Moderate", "Moderate", None, "High", "High"],
        }
    )

    spec = _outcome_distribution_steps_v2_spec(
        data,
        state_order=["Low", "Moderate", "High"],
    )

    assert spec["schemaVersion"] == "2.0"
    assert spec["type"] == "outcome_distribution"
    assert spec["xAxisKind"] == "step"
    assert spec["stateOrder"] == ["Low", "Moderate", "High"]

    # Calculate independent reference using Polars
    valid_data = data.filter(pl.col("state").is_not_null())
    unique_steps = sorted(valid_data["step"].unique().to_list())

    distributions = spec["stateDistributions"]
    assert len(distributions) == len(unique_steps)

    expected_steps = [0, 1, 3]
    for i, step in enumerate(expected_steps):
        dist = distributions[i]
        assert dist["evaluationId"] == "evaluation-1"
        assert dist["horizon"] == step
        assert dist["estimator"] == "raw"
        assert dist["estimateOrigin"] == "observed_step"

        step_df = valid_data.filter(pl.col("step") == step)
        expected_n = len(step_df)

        actual_counts = {st["stateId"]: st["count"] for st in dist["states"]}
        actual_total = sum(actual_counts.values())
        assert actual_total == expected_n

        ref_counts = step_df.group_by("state").len()
        ref_map = {row["state"]: row["len"] for row in ref_counts.iter_rows(named=True)}

        for sid in ["Low", "Moderate", "High"]:
            assert actual_counts[sid] == ref_map.get(sid, 0)

    # Step 0: Low=1, Moderate=1, High=1 (N=3)
    s0 = distributions[0]
    c0 = {st["stateId"]: st["count"] for st in s0["states"]}
    assert c0 == {"Low": 1, "Moderate": 1, "High": 1}

    # Step 1: Low=0, Moderate=2, High=0 (N=2, C is null and excluded; proportion Moderate = 2/2 = 1.0)
    s1 = distributions[1]
    c1 = {st["stateId"]: st["count"] for st in s1["states"]}
    assert c1 == {"Low": 0, "Moderate": 2, "High": 0}

    # Step 3: Low=0, Moderate=0, High=2 (N=2; proportion High = 2/2 = 1.0)
    s3 = distributions[2]
    c3 = {st["stateId"]: st["count"] for st in s3["states"]}
    assert c3 == {"Low": 0, "Moderate": 0, "High": 2}


def test_unbalanced_panels_and_late_entry():
    data = pl.DataFrame(
        {
            "subject": ["A", "B", "C", "A", "C"],
            "step": [0, 0, 1, 2, 2],
            "state": ["S1", "S2", "S1", "S2", "S2"],
        }
    )
    spec = _outcome_distribution_steps_v2_spec(data)
    steps = [sd["horizon"] for sd in spec["stateDistributions"]]
    assert steps == [0, 1, 2]

    # At step 1, only C is observed (N=1)
    s1 = spec["stateDistributions"][1]
    counts1 = {st["stateId"]: st["count"] for st in s1["states"]}
    assert counts1["S1"] == 1
    assert counts1["S2"] == 0


def test_intermittent_missingness():
    data = pl.DataFrame(
        {
            "subject": ["A", "A", "A"],
            "step": [0, 1, 2],
            "state": ["OK", None, "OK"],
        }
    )
    spec = _outcome_distribution_steps_v2_spec(data)
    steps = [sd["horizon"] for sd in spec["stateDistributions"]]
    assert steps == [0, 2]


def test_all_missing_steps_and_all_missing_df_raises():
    data_all_missing = pl.DataFrame(
        {
            "subject": ["A", "B"],
            "step": [0, 1],
            "state": [None, None],
        }
    )
    with pytest.raises(ValueError, match="No valid observed steps remain"):
        _outcome_distribution_steps_v2_spec(data_all_missing)


def test_duplicate_subject_step_records_raises():
    data = pl.DataFrame(
        {
            "subject": ["A", "A"],
            "step": [0, 0],
            "state": ["S1", "S2"],
        }
    )
    with pytest.raises(ValueError, match="Duplicate subject-step pairs"):
        _outcome_distribution_steps_v2_spec(data)

    data_dup_null = pl.DataFrame(
        {
            "subject": ["A", "A"],
            "step": [0, 0],
            "state": ["S1", None],
        }
    )
    with pytest.raises(ValueError, match="Duplicate subject-step pairs"):
        _outcome_distribution_steps_v2_spec(data_dup_null)


def test_nonconsecutive_steps_and_no_synthetic_steps():
    data = pl.DataFrame(
        {
            "subject": ["A", "B"],
            "step": [10, 50],
            "state": ["X", "Y"],
        }
    )
    spec = _outcome_distribution_steps_v2_spec(data)
    horizons = [sd["horizon"] for sd in spec["stateDistributions"]]
    assert horizons == [10, 50]


def test_numeric_and_string_state_identifiers():
    data_num = pl.DataFrame(
        {
            "subject": ["A", "B", "C"],
            "step": [0, 0, 0],
            "state": [1, 2, 3],
        }
    )
    spec_num = _outcome_distribution_steps_v2_spec(data_num)
    assert spec_num["stateOrder"] == ["1", "2", "3"]

    data_str = pl.DataFrame(
        {
            "subject": ["A", "B"],
            "step": [0, 0],
            "state": ["beta", "alpha"],
        }
    )
    spec_str = _outcome_distribution_steps_v2_spec(data_str)
    assert spec_str["stateOrder"] == ["alpha", "beta"]


def test_polars_enum_categorical_ordering():
    enum_dtype = pl.Enum(["High", "Moderate", "Low"])
    data = pl.DataFrame(
        {
            "subject": ["A", "B", "C"],
            "step": [0, 0, 0],
            "state": pl.Series(["Low", "High", "Moderate"], dtype=enum_dtype),
        }
    )
    spec = _outcome_distribution_steps_v2_spec(data)
    assert spec["stateOrder"] == ["High", "Moderate", "Low"]


def test_invalid_ordering_declarations():
    data = pl.DataFrame(
        {
            "subject": ["A", "B"],
            "step": [0, 0],
            "state": ["Low", "High"],
        }
    )
    # Duplicate in state_order
    with pytest.raises(ValueError, match="duplicate values"):
        _outcome_distribution_steps_v2_spec(data, state_order=["Low", "Low", "High"])

    # Unobserved state in state_order
    with pytest.raises(ValueError, match="does not match observed states exactly"):
        _outcome_distribution_steps_v2_spec(data, state_order=["Low", "High", "Medium"])

    # Incomplete state_order
    with pytest.raises(ValueError, match="does not match observed states exactly"):
        _outcome_distribution_steps_v2_spec(data, state_order=["Low"])


def test_invalid_steps():
    # Negative step
    data_neg = pl.DataFrame({"subject": ["A"], "step": [-1], "state": ["OK"]})
    with pytest.raises(ValueError, match="non-negative"):
        _outcome_distribution_steps_v2_spec(data_neg)

    # Fractional step
    data_frac = pl.DataFrame({"subject": ["A"], "step": [1.5], "state": ["OK"]})
    with pytest.raises(ValueError, match="integer-valued"):
        _outcome_distribution_steps_v2_spec(data_frac)

    # Null step
    data_null_step = pl.DataFrame({"subject": ["A"], "step": [None], "state": ["OK"]})
    with pytest.raises(ValueError, match="contains null values"):
        _outcome_distribution_steps_v2_spec(data_null_step)


def test_null_subject():
    data_null_sub = pl.DataFrame({"subject": [None], "step": [0], "state": ["OK"]})
    with pytest.raises(ValueError, match="contains null values"):
        _outcome_distribution_steps_v2_spec(data_null_sub)


def test_state_value_collision():
    # If state column contains int 1 and string '1', they serialize to same stateId '1'
    data = pl.DataFrame({
        "subject": ["A", "B"],
        "step": [0, 0],
        "state": pl.Series("state", [1, "1"], dtype=pl.Object),
    })
    with pytest.raises(ValueError, match="State value collision"):
        _outcome_distribution_steps_v2_spec(data)


def test_public_create_reals_distribution_steps():
    data = pl.DataFrame(
        {
            "subject": ["A", "B"],
            "step": [0, 0],
            "state": ["S1", "S2"],
        }
    )
    chart = create_reals_distribution_steps(data)
    assert hasattr(chart, "spec")
    assert chart.spec["type"] == "outcome_distribution"
    assert chart.spec["xAxisKind"] == "step"

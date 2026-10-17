"""Tests for create_reals_distribution_steps and discrete-step outcome distributions."""

import polars as pl
import pytest
import rtichoke
from rtichoke.reals_distribution import create_reals_distribution_steps


def test_public_export():
    assert hasattr(rtichoke, "create_reals_distribution_steps")
    assert "create_reals_distribution_steps" in rtichoke.__all__


def test_input_validation():
    # Non-DataFrame input
    with pytest.raises(TypeError, match="must be a Polars DataFrame"):
        create_reals_distribution_steps([{"id": 1, "step": 0, "reals": "Mild"}])  # type: ignore

    df = pl.DataFrame({"id": [1], "step": [0], "reals": ["Mild"]})

    # Unsupported renderer
    with pytest.raises(ValueError, match="Unsupported renderer"):
        create_reals_distribution_steps(df, renderer="plotly")  # type: ignore

    # Invalid censoring_heuristic
    with pytest.raises(ValueError, match="Invalid censoring_heuristic"):
        create_reals_distribution_steps(df, censoring_heuristic="invalid")  # type: ignore

    # Invalid reals_scale
    with pytest.raises(ValueError, match="Invalid reals_scale"):
        create_reals_distribution_steps(df, reals_scale="invalid")  # type: ignore

    # Missing column
    with pytest.raises(ValueError, match="Missing required columns"):
        create_reals_distribution_steps(df, reals_col="missing")

    # Missing/null IDs
    df_null_id = pl.DataFrame(
        {"id": [None, 1], "step": [0, 1], "reals": ["Mild", "Moderate"]}
    )
    with pytest.raises(ValueError, match="contains missing/null values"):
        create_reals_distribution_steps(df_null_id)

    # Missing/null steps
    df_null_step = pl.DataFrame(
        {"id": [1, 2], "step": [0, None], "reals": ["Mild", "Moderate"]}
    )
    with pytest.raises(ValueError, match="contains missing/null values"):
        create_reals_distribution_steps(df_null_step)

    # Negative steps
    df_neg_step = pl.DataFrame({"id": [1], "step": [-1], "reals": ["Mild"]})
    with pytest.raises(ValueError, match="negative step values"):
        create_reals_distribution_steps(df_neg_step)

    # Non-integer step values
    df_float_step = pl.DataFrame({"id": [1], "step": [1.5], "reals": ["Mild"]})
    with pytest.raises(ValueError, match="non-negative integer values"):
        create_reals_distribution_steps(df_float_step)

    # Duplicate subject-step pairs
    df_dup = pl.DataFrame({"id": [1, 1], "step": [0, 0], "reals": ["Mild", "Moderate"]})
    with pytest.raises(ValueError, match="Duplicate records found"):
        create_reals_distribution_steps(df_dup)


def test_reals_order_and_scale_resolution():
    # 1. Explicit reals_order
    df = pl.DataFrame(
        {"id": [1, 2, 3], "step": [0, 0, 0], "reals": ["Moderate", "Mild", "None"]}
    )
    chart = create_reals_distribution_steps(
        df, reals_order=["None", "Mild", "Moderate"]
    )
    assert (
        chart.spec["stateScale"] == "ordinal"
    )  # Auto resolved to ordinal because reals_order is explicit
    states = chart.spec["stateDistributions"][0]["states"]
    assert [s["stateId"] for s in states] == ["None", "Mild", "Moderate"]

    # 2. Explicit reals_scale
    chart_nom = create_reals_distribution_steps(
        df, reals_order=["None", "Mild", "Moderate"], reals_scale="nominal"
    )
    assert chart_nom.spec["stateScale"] == "nominal"

    # 3. Polars Enum
    enum_dtype = pl.Enum(["None", "Mild", "Moderate"])
    df_enum = pl.DataFrame(
        {
            "id": [1, 2, 3],
            "step": [0, 0, 0],
            "reals": pl.Series(["Moderate", "Mild", "None"], dtype=enum_dtype),
        }
    )
    chart_enum = create_reals_distribution_steps(df_enum)
    assert chart_enum.spec["stateScale"] == "ordinal"
    states_enum = chart_enum.spec["stateDistributions"][0]["states"]
    assert [s["stateId"] for s in states_enum] == ["None", "Mild", "Moderate"]

    # 4. Numeric outcome categories
    df_num = pl.DataFrame({"id": [1, 2, 3], "step": [0, 0, 0], "reals": [2, 0, 1]})
    chart_num = create_reals_distribution_steps(df_num)
    assert chart_num.spec["stateScale"] == "ordinal"
    states_num = chart_num.spec["stateDistributions"][0]["states"]
    assert [s["stateId"] for s in states_num] == ["0", "1", "2"]

    # 5. String/categorical nominal default
    df_str = pl.DataFrame({"id": [1, 2], "step": [0, 0], "reals": ["Red", "Blue"]})
    chart_str = create_reals_distribution_steps(df_str)
    assert chart_str.spec["stateScale"] == "nominal"


def test_complete_panel_raw_and_aj():
    df = pl.DataFrame(
        {
            "patient_id": [1, 1, 1, 2, 2, 2],
            "visit": [0, 1, 2, 0, 1, 2],
            "severity": ["None", "Mild", "Moderate", "None", "Mild", "Mild"],
        }
    )

    # Raw
    chart_raw = create_reals_distribution_steps(
        data=df,
        id_col="patient_id",
        step_col="visit",
        reals_col="severity",
        reals_order=["None", "Mild", "Moderate"],
        censoring_heuristic="excluded",
    )

    spec_raw = chart_raw.spec
    assert spec_raw["schemaVersion"] == "2.0"
    assert spec_raw["type"] == "outcome_distribution"
    assert spec_raw["title"] == "Outcome Distribution"
    assert spec_raw["evaluations"] == [
        {"id": "evaluation-1", "population": "__shared_population__"}
    ]

    dist_raw = spec_raw["stateDistributions"]
    assert len(dist_raw) == 3

    # Step 0: 2 None, 0 Mild, 0 Moderate
    assert dist_raw[0]["horizon"] == 0
    assert dist_raw[0]["estimator"] == "raw"
    assert dist_raw[0]["estimateOrigin"] == "observed_step"
    assert dist_raw[0]["states"] == [
        {"stateId": "None", "label": "None", "count": 2},
        {"stateId": "Mild", "label": "Mild", "count": 0},
        {"stateId": "Moderate", "label": "Moderate", "count": 0},
    ]

    # Step 1: 0 None, 2 Mild, 0 Moderate
    assert dist_raw[1]["horizon"] == 1
    assert dist_raw[1]["states"] == [
        {"stateId": "None", "label": "None", "count": 0},
        {"stateId": "Mild", "label": "Mild", "count": 2},
        {"stateId": "Moderate", "label": "Moderate", "count": 0},
    ]

    # Step 2: 0 None, 1 Mild, 1 Moderate
    assert dist_raw[2]["horizon"] == 2
    assert dist_raw[2]["states"] == [
        {"stateId": "None", "label": "None", "count": 0},
        {"stateId": "Mild", "label": "Mild", "count": 1},
        {"stateId": "Moderate", "label": "Moderate", "count": 1},
    ]

    # AJ
    chart_aj = create_reals_distribution_steps(
        data=df,
        id_col="patient_id",
        step_col="visit",
        reals_col="severity",
        reals_order=["None", "Mild", "Moderate"],
        censoring_heuristic="adjusted",
    )

    spec_aj = chart_aj.spec
    dist_aj = spec_aj["stateDistributions"]
    assert len(dist_aj) == 3

    # Step 0: None 1.0, Mild 0.0, Moderate 0.0
    assert dist_aj[0]["horizon"] == 0
    assert dist_aj[0]["estimator"] == "aalen_johansen"
    assert dist_aj[0]["estimateOrigin"] == "observed_step"
    assert dist_aj[0]["states"] == [
        {"stateId": "None", "label": "None", "estimate": 1.0},
        {"stateId": "Mild", "label": "Mild", "estimate": 0.0},
        {"stateId": "Moderate", "label": "Moderate", "estimate": 0.0},
    ]

    # Step 2: None 0.0, Mild 0.5, Moderate 0.5
    assert dist_aj[2]["horizon"] == 2
    assert dist_aj[2]["states"] == [
        {"stateId": "None", "label": "None", "estimate": 0.0},
        {"stateId": "Mild", "label": "Mild", "estimate": 0.5},
        {"stateId": "Moderate", "label": "Moderate", "estimate": 0.5},
    ]


def test_missing_reals_excluded():
    # Intermittent missing outcome at step 1 for patient 2
    df = pl.DataFrame(
        {
            "id": [1, 1, 1, 2, 2, 2],
            "step": [0, 1, 2, 0, 1, 2],
            "reals": ["None", "Mild", "Moderate", "None", None, "Mild"],
        }
    )

    chart = create_reals_distribution_steps(
        df, reals_order=["None", "Mild", "Moderate"], censoring_heuristic="excluded"
    )
    dist = chart.spec["stateDistributions"]

    # Step 1 has 1 observed outcome (Mild) for subject 1, subject 2 missing. Count = 1 Mild.
    assert dist[1]["horizon"] == 1
    assert dist[1]["states"] == [
        {"stateId": "None", "label": "None", "count": 0},
        {"stateId": "Mild", "label": "Mild", "count": 1},
        {"stateId": "Moderate", "label": "Moderate", "count": 0},
    ]


def test_nonconsecutive_steps_and_no_step_0():
    df = pl.DataFrame(
        {"id": [1, 1, 2, 2], "step": [5, 10, 5, 10], "reals": ["A", "B", "A", "A"]}
    )

    chart = create_reals_distribution_steps(df, censoring_heuristic="excluded")
    dist = chart.spec["stateDistributions"]
    assert len(dist) == 2
    assert dist[0]["horizon"] == 5
    assert dist[1]["horizon"] == 10


def test_zero_count_category_preservation():
    df = pl.DataFrame({"id": [1, 2], "step": [0, 0], "reals": ["Mild", "Mild"]})

    chart = create_reals_distribution_steps(
        df, reals_order=["None", "Mild", "Severe"], censoring_heuristic="excluded"
    )
    states = chart.spec["stateDistributions"][0]["states"]
    assert states == [
        {"stateId": "None", "label": "None", "count": 0},
        {"stateId": "Mild", "label": "Mild", "count": 2},
        {"stateId": "Severe", "label": "Severe", "count": 0},
    ]

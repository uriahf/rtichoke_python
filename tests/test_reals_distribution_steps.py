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

    # Observed outcomes not in declared reals_order
    df_unrecognized = pl.DataFrame(
        {"id": [1, 2], "step": [0, 0], "reals": ["Mild", "Severe"]}
    )
    with pytest.raises(
        ValueError, match="Observed outcomes contain values not present in declared"
    ):
        create_reals_distribution_steps(df_unrecognized, reals_order=["None", "Mild"])


def test_reals_order_and_scale_resolution():
    # 1. Explicit reals_order on ordinary strings remains nominal under auto
    df = pl.DataFrame(
        {"id": [1, 2, 3], "step": [0, 0, 0], "reals": ["Blue", "Green", "Red"]}
    )
    chart = create_reals_distribution_steps(df, reals_order=["Red", "Green", "Blue"])
    assert (
        chart.spec["stateScale"] == "nominal"
    )  # Explicit reals_order alone does NOT imply ordinal scale
    states = chart.spec["stateDistributions"][0]["states"]
    assert [s["stateId"] for s in states] == ["Red", "Green", "Blue"]

    # 2. Explicit reals_scale="ordinal" overrides default nominal
    chart_ord = create_reals_distribution_steps(
        df, reals_order=["Red", "Green", "Blue"], reals_scale="ordinal"
    )
    assert chart_ord.spec["stateScale"] == "ordinal"

    # 3. Polars Enum auto-resolves to ordinal
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

    # 4. Numeric outcome categories auto-resolve to ordinal
    df_num = pl.DataFrame({"id": [1, 2, 3], "step": [0, 0, 0], "reals": [2, 0, 1]})
    chart_num = create_reals_distribution_steps(df_num)
    assert chart_num.spec["stateScale"] == "ordinal"
    states_num = chart_num.spec["stateDistributions"][0]["states"]
    assert [s["stateId"] for s in states_num] == ["0", "1", "2"]


def test_mass_conservation_and_sum_checks():
    df = pl.DataFrame(
        {
            "id": [1, 1, 1, 2, 2, 2, 3, 3, 3],
            "step": [0, 1, 2, 0, 1, 2, 0, 1, 2],
            "reals": [
                "None",
                "Mild",
                "Moderate",
                "None",
                "Mild",
                "Mild",
                "None",
                "None",
                "Mild",
            ],
        }
    )

    # Raw counts sum check
    chart_raw = create_reals_distribution_steps(
        data=df,
        reals_order=["None", "Mild", "Moderate"],
        censoring_heuristic="excluded",
    )
    for dist in chart_raw.spec["stateDistributions"]:
        total_counts = sum(s["count"] for s in dist["states"])
        assert total_counts == 3

    # AJ probabilities sum to 1.0 check
    chart_aj = create_reals_distribution_steps(
        data=df,
        reals_order=["None", "Mild", "Moderate"],
        censoring_heuristic="adjusted",
    )
    for dist in chart_aj.spec["stateDistributions"]:
        total_prob = sum(s["estimate"] for s in dist["states"])
        assert pytest.approx(total_prob, abs=1e-6) == 1.0


def test_aj_scenarios_and_independent_validation():
    # 1. Complete panel
    df_complete = pl.DataFrame(
        {
            "id": [1, 1, 1, 2, 2, 2],
            "step": [0, 1, 2, 0, 1, 2],
            "reals": ["None", "Mild", "Moderate", "None", "Mild", "Mild"],
        }
    )
    chart_complete = create_reals_distribution_steps(
        df_complete,
        reals_order=["None", "Mild", "Moderate"],
        censoring_heuristic="adjusted",
    )
    dist_c = chart_complete.spec["stateDistributions"]
    # Step 0: None=1.0, Mild=0.0, Moderate=0.0
    assert dist_c[0]["states"] == [
        {"stateId": "None", "label": "None", "estimate": 1.0},
        {"stateId": "Mild", "label": "Mild", "estimate": 0.0},
        {"stateId": "Moderate", "label": "Moderate", "estimate": 0.0},
    ]
    # Step 1: None=0.0, Mild=1.0, Moderate=0.0
    assert dist_c[1]["states"] == [
        {"stateId": "None", "label": "None", "estimate": 0.0},
        {"stateId": "Mild", "label": "Mild", "estimate": 1.0},
        {"stateId": "Moderate", "label": "Moderate", "estimate": 0.0},
    ]
    # Step 2: None=0.0, Mild=0.5, Moderate=0.5
    assert dist_c[2]["states"] == [
        {"stateId": "None", "label": "None", "estimate": 0.0},
        {"stateId": "Mild", "label": "Mild", "estimate": 0.5},
        {"stateId": "Moderate", "label": "Moderate", "estimate": 0.5},
    ]

    # 2. Terminal dropout / incomplete panel
    # Patient 2 drops out after step 1
    df_dropout = pl.DataFrame(
        {
            "id": [1, 1, 1, 2, 2],
            "step": [0, 1, 2, 0, 1],
            "reals": ["None", "Mild", "Moderate", "None", "Mild"],
        }
    )
    chart_dropout = create_reals_distribution_steps(
        df_dropout,
        reals_order=["None", "Mild", "Moderate"],
        censoring_heuristic="adjusted",
    )
    dist_d = chart_dropout.spec["stateDistributions"]
    # Step 2: Subject 1 moved Mild -> Moderate. Subject 2 dropped out while in Mild.
    # AJ occupancy at step 2: None=0.0, Mild=0.0, Moderate=1.0
    assert dist_d[2]["states"] == [
        {"stateId": "None", "label": "None", "estimate": 0.0},
        {"stateId": "Mild", "label": "Mild", "estimate": 0.0},
        {"stateId": "Moderate", "label": "Moderate", "estimate": 1.0},
    ]

    # 3. Intermittent missing outcomes under 'excluded' vs 'adjusted'
    df_intermittent = pl.DataFrame(
        {
            "id": [1, 1, 1, 2, 2, 2],
            "step": [0, 1, 2, 0, 1, 2],
            "reals": ["None", "Mild", "Moderate", "None", None, "Mild"],
        }
    )
    chart_excl = create_reals_distribution_steps(
        df_intermittent,
        reals_order=["None", "Mild", "Moderate"],
        censoring_heuristic="excluded",
    )
    # Under 'excluded', at step 1 only subject 1 is observed (Mild), subject 2 is excluded from denominator.
    assert chart_excl.spec["stateDistributions"][1]["states"] == [
        {"stateId": "None", "label": "None", "count": 0},
        {"stateId": "Mild", "label": "Mild", "count": 1},
        {"stateId": "Moderate", "label": "Moderate", "count": 0},
    ]


def test_baseline_only_subjects():
    # Patient 2 observed only at step 0
    df = pl.DataFrame(
        {
            "id": [1, 1, 2],
            "step": [0, 1, 0],
            "reals": ["None", "Mild", "None"],
        }
    )
    chart = create_reals_distribution_steps(
        df,
        reals_order=["None", "Mild"],
        censoring_heuristic="adjusted",
    )
    dist = chart.spec["stateDistributions"]
    assert len(dist) == 2
    # Step 0: None=1.0, Mild=0.0
    assert dist[0]["states"] == [
        {"stateId": "None", "label": "None", "estimate": 1.0},
        {"stateId": "Mild", "label": "Mild", "estimate": 0.0},
    ]
    # Step 1: Subject 2 dropped out from None. Subject 1 transitioned None -> Mild.
    # AJ occupancy at step 1: None=0.0, Mild=1.0
    assert dist[1]["states"] == [
        {"stateId": "None", "label": "None", "estimate": 0.0},
        {"stateId": "Mild", "label": "Mild", "estimate": 1.0},
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

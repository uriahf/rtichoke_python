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


def test_independent_multistate_aj_reference_with_right_censoring():
    """Independent multi-state Aalen-Johansen verification with right-censoring.

    Independent reference problem formulation (equivalent to R survival::survfit multistate AJ):
    Consider a 3-state scheduled discrete Markov panel observed at steps 0, 1, and 2 with states A, B, and C.
    4 subjects are enrolled at baseline (step 0):
      Subject 1: step 0 -> A, step 1 -> B, step 2 -> C
      Subject 2: step 0 -> A, step 1 -> A, step 2 -> B
      Subject 3: step 0 -> B, step 1 -> B, step 2 -> Censored / dropped out before step 2
      Subject 4: step 0 -> B, step 1 -> C, step 2 -> C

    Step 0 baseline distribution:
      n=4: 2 in A, 2 in B, 0 in C
      p0 = [P(A)=2/4=0.5, P(B)=2/4=0.5, P(C)=0/4=0.0]

    Transitions 0 -> 1:
      From A at 0 (subjects 1, 2):
        Subject 1 transitions A -> B
        Subject 2 transitions A -> A
        P(A -> A) = 1/2 = 0.5, P(A -> B) = 1/2 = 0.5, P(A -> C) = 0.0
      From B at 0 (subjects 3, 4):
        Subject 3 transitions B -> B
        Subject 4 transitions B -> C
        P(B -> A) = 0.0, P(B -> B) = 1/2 = 0.5, P(B -> C) = 1/2 = 0.5
      Transition matrix T(0 -> 1):
        [[0.5, 0.5, 0.0],
         [0.0, 0.5, 0.5],
         [0.0, 0.0, 1.0]]
      Occupancy at step 1:
        p1 = p0 * T(0 -> 1) = [0.5*0.5, 0.5*0.5 + 0.5*0.5, 0.5*0.5] = [0.25, 0.50, 0.25]

    Transitions 1 -> 2 with Right-Censoring:
      Subjects under observation at step 1:
        Subject 1 in B
        Subject 2 in A
        Subject 3 in B (drops out after step 1; censored before step 2)
        Subject 4 in C
      At-risk risk set Y_i(2) for step 2:
        From A at 1: Subject 2 remains observed -> transitions A -> B. Y_A(2) = 1.
          P(A -> B) = 1/1 = 1.0, other transitions = 0.0.
        From B at 1: Subject 1 remains observed -> transitions B -> C.
          Subject 3 is right-censored before step 2 and excluded from step 2 transitions.
          Y_B(2) = 1 (Subject 1 only).
          P(B -> C) = 1/1 = 1.0, other transitions = 0.0.
        From C at 1: Subject 4 remains observed -> transitions C -> C. Y_C(2) = 1.
          P(C -> C) = 1/1 = 1.0.
      Transition matrix T(1 -> 2):
        [[0.0, 1.0, 0.0],
         [0.0, 0.0, 1.0],
         [0.0, 0.0, 1.0]]
      Occupancy at step 2:
        p2 = p1 * T(1 -> 2)
           = [0.25*0.0, 0.25*1.0, 0.50*1.0 + 0.25*1.0]
           = [0.0, 0.25, 0.75]

    Note contrast with crude/naive proportions at step 2:
      Observed subjects at step 2: Subject 1 (C), Subject 2 (B), Subject 4 (C). Total = 3.
      Crude proportions: P(A)=0/3=0.0, P(B)=1/3≈0.3333, P(C)=2/3≈0.6667.
      The Aalen-Johansen adjusted occupancy properly accounts for the censoring of Subject 3
      from state B, yielding P(B)=0.25 and P(C)=0.75 != 0.6667.
    """
    df = pl.DataFrame(
        {
            "id": [1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 4],
            "step": [0, 1, 2, 0, 1, 2, 0, 1, 0, 1, 2],
            "reals": ["A", "B", "C", "A", "A", "B", "B", "B", "B", "C", "C"],
        }
    )

    chart = create_reals_distribution_steps(
        df,
        reals_order=["A", "B", "C"],
        reals_scale="nominal",
        censoring_heuristic="adjusted",
    )

    dists = chart.spec["stateDistributions"]
    assert len(dists) == 3

    # Step 0: [0.5, 0.5, 0.0]
    p0 = {s["stateId"]: s["estimate"] for s in dists[0]["states"]}
    assert pytest.approx(p0["A"], abs=1e-9) == 0.50
    assert pytest.approx(p0["B"], abs=1e-9) == 0.50
    assert pytest.approx(p0["C"], abs=1e-9) == 0.00

    # Step 1: [0.25, 0.50, 0.25]
    p1 = {s["stateId"]: s["estimate"] for s in dists[1]["states"]}
    assert pytest.approx(p1["A"], abs=1e-9) == 0.25
    assert pytest.approx(p1["B"], abs=1e-9) == 0.50
    assert pytest.approx(p1["C"], abs=1e-9) == 0.25

    # Step 2: [0.0, 0.25, 0.75]
    p2 = {s["stateId"]: s["estimate"] for s in dists[2]["states"]}
    assert pytest.approx(p2["A"], abs=1e-9) == 0.00
    assert pytest.approx(p2["B"], abs=1e-9) == 0.25
    assert pytest.approx(p2["C"], abs=1e-9) == 0.75

    # Verify that crude proportions under excluded differ from AJ estimates at step 2
    chart_raw = create_reals_distribution_steps(
        df,
        reals_order=["A", "B", "C"],
        reals_scale="nominal",
        censoring_heuristic="excluded",
    )
    raw_counts = {
        s["stateId"]: s["count"]
        for s in chart_raw.spec["stateDistributions"][2]["states"]
    }
    assert raw_counts == {"A": 0, "B": 1, "C": 2}
    total_raw = sum(raw_counts.values())  # 3
    raw_p_B = raw_counts["B"] / total_raw  # 1/3 = 0.3333 != 0.25
    raw_p_C = raw_counts["C"] / total_raw  # 2/3 = 0.6667 != 0.75
    assert abs(raw_p_B - p2["B"]) > 0.05
    assert abs(raw_p_C - p2["C"]) > 0.05

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


def test_step_with_all_missing_outcomes_is_preserved_in_excluded():
    # Step 1 has only missing outcomes; it must be preserved with 0 counts for all states
    df = pl.DataFrame(
        {
            "id": [1, 1, 1, 2, 2, 2],
            "step": [0, 1, 2, 0, 1, 2],
            "reals": ["None", None, "Moderate", "None", None, "Mild"],
        }
    )
    chart = create_reals_distribution_steps(
        df,
        reals_order=["None", "Mild", "Moderate"],
        censoring_heuristic="excluded",
    )
    dists = chart.spec["stateDistributions"]
    assert len(dists) == 3
    assert dists[1]["horizon"] == 1
    assert dists[1]["states"] == [
        {"stateId": "None", "label": "None", "count": 0},
        {"stateId": "Mild", "label": "Mild", "count": 0},
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


def test_write_html_outcome_distribution(tmp_path):
    df = pl.DataFrame(
        {
            "id": [1, 1, 2, 2],
            "step": [0, 1, 0, 1],
            "reals": ["None", "Mild", "None", "Moderate"],
        }
    )

    # 1. Censoring heuristic: excluded
    chart_excluded = create_reals_distribution_steps(
        df,
        reals_order=["None", "Mild", "Moderate"],
        censoring_heuristic="excluded",
    )
    out_excluded = chart_excluded.write_html(tmp_path / "excluded.html")
    assert out_excluded.exists()
    content_excluded = out_excluded.read_text(encoding="utf-8")
    assert "renderOutcomeDistribution" in content_excluded
    assert "rtichoke-viz.js" in content_excluded
    assert "outcome_distribution" in content_excluded

    # 2. Censoring heuristic: adjusted
    chart_adjusted = create_reals_distribution_steps(
        df,
        reals_order=["None", "Mild", "Moderate"],
        censoring_heuristic="adjusted",
    )
    out_adjusted = chart_adjusted.write_html(tmp_path / "adjusted.html")
    assert out_adjusted.exists()
    content_adjusted = out_adjusted.read_text(encoding="utf-8")
    assert "renderOutcomeDistribution" in content_adjusted
    assert "rtichoke-viz.js" in content_adjusted
    assert "outcome_distribution" in content_adjusted

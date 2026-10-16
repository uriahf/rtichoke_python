import json
from pathlib import Path

import polarstate
import polars as pl
import pytest

# Ensure discrete-step polarstate functions are available for testing
if not hasattr(polarstate, "prepare_transition_table_steps"):
    import importlib.util

    ps_file = getattr(polarstate, "__file__", None)
    step_module_path = None
    if ps_file:
        step_module_path = Path(ps_file).parent / "transition_steps.py"

    if not step_module_path or not step_module_path.exists():
        dev_path = (
            Path(__file__).parents[2]
            / "polarstate"
            / "src"
            / "polarstate"
            / "transition_steps.py"
        )
        if dev_path.exists():
            step_module_path = dev_path
        else:
            step_module_path = Path(
                "/home/jules/polarstate/src/polarstate/transition_steps.py"
            )

    if step_module_path and step_module_path.exists():
        spec = importlib.util.spec_from_file_location("ps_steps", step_module_path)
        if spec and spec.loader:
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            polarstate.prepare_transition_table_steps = getattr(
                mod, "prepare_transition_table_steps", None
            )
            polarstate._predict_transition_steps = getattr(
                mod, "_predict_transition_steps", None
            )
            polarstate.TransitionTableSteps = getattr(mod, "TransitionTableSteps", None)

from rtichoke.reals_distribution import (
    create_reals_distribution_steps,
    create_reals_distribution_times,
)


# ------------------------------------------------------------------------------
# Fixtures & Verified Reference Results
# ------------------------------------------------------------------------------

COMPLETE_PANEL_1 = pl.DataFrame(
    {
        "id": [1, 1, 1, 2, 2, 2, 3, 3, 3],
        "step": [0, 1, 2, 0, 1, 2, 0, 1, 2],
        "reals": [1, 1, 2, 1, 2, 2, 1, 1, 1],
    }
)

COMPLETE_PANEL_2 = pl.DataFrame(
    {
        "id": [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
        "step": [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2],
        "reals": [1, 2, 3, 1, 1, 2, 2, 3, 2, 3, 2, 1],
    }
)

STRING_PANEL = pl.DataFrame(
    {
        "id": [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5],
        "step": [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2],
        "reals": [
            "A",
            "B",
            "C",
            "B",
            "B",
            "A",
            "C",
            "A",
            "B",
            "A",
            "A",
            "C",
            "B",
            "C",
            "C",
        ],
    }
)

DROPOUT_PANEL_7 = pl.DataFrame(
    {
        "id": ["A", "A", "A", "B", "B", "B", "C", "C", "D", "D", "D", "E", "E", "F"],
        "step": [0, 1, 2, 0, 1, 2, 0, 1, 0, 1, 2, 0, 1, 0],
        "reals": [1, 1, 2, 1, 2, 2, 1, 1, 2, 2, 1, 2, 1, 2],
    }
)

BASELINE_ONLY_PANEL_9 = pl.DataFrame(
    {
        "id": ["A", "B", "B", "C", "C"],
        "step": [0, 0, 1, 0, 1],
        "reals": [1, 1, 2, 2, 2],
    }
)


def test_create_reals_distribution_steps_returns_browser_chart_excluded():
    chart = create_reals_distribution_steps(
        COMPLETE_PANEL_1, censoring_heuristic="excluded"
    )
    assert hasattr(chart, "spec")
    spec = chart.spec

    assert spec["schemaVersion"] == "2.0"
    assert spec["type"] == "outcome_distribution"
    assert spec["title"] == "Outcome Distribution"
    assert spec["evaluations"] == [
        {"id": "evaluation-1", "population": "__shared_population__"}
    ]

    sds = spec["stateDistributions"]
    assert len(sds) == 3  # steps 0, 1, 2

    for sd in sds:
        assert sd["evaluationId"] == "evaluation-1"
        assert sd["estimator"] == "raw"
        assert sd["estimateOrigin"] == "observed_step"
        assert isinstance(sd["horizon"], int)
        for state in sd["states"]:
            assert "count" in state
            assert isinstance(state["count"], int)


def test_create_reals_distribution_steps_returns_browser_chart_adjusted():
    chart = create_reals_distribution_steps(
        COMPLETE_PANEL_1, censoring_heuristic="adjusted"
    )
    assert hasattr(chart, "spec")
    spec = chart.spec

    assert spec["schemaVersion"] == "2.0"
    assert spec["type"] == "outcome_distribution"
    assert spec["title"] == "Outcome Distribution"

    sds = spec["stateDistributions"]
    assert len(sds) == 3  # steps 0, 1, 2

    for sd in sds:
        assert sd["evaluationId"] == "evaluation-1"
        assert sd["estimator"] == "aalen_johansen"
        assert sd["estimateOrigin"] == "observed_step"
        assert isinstance(sd["horizon"], int)
        for state in sd["states"]:
            assert "estimate" in state
            assert isinstance(state["estimate"], float)


@pytest.mark.parametrize("df", [COMPLETE_PANEL_1, COMPLETE_PANEL_2, STRING_PANEL])
def test_complete_panel_proportions_equal_aj_occupancy(df: pl.DataFrame):
    chart_ex = create_reals_distribution_steps(df, censoring_heuristic="excluded")
    chart_adj = create_reals_distribution_steps(df, censoring_heuristic="adjusted")

    sds_ex = chart_ex.spec["stateDistributions"]
    sds_adj = chart_adj.spec["stateDistributions"]

    assert len(sds_ex) == len(sds_adj)

    for sd_ex, sd_adj in zip(sds_ex, sds_adj):
        assert sd_ex["horizon"] == sd_adj["horizon"]
        counts = {st["stateId"]: st["count"] for st in sd_ex["states"]}
        total_obs = sum(counts.values())

        for st_ex, st_adj in zip(sd_ex["states"], sd_adj["states"]):
            assert st_ex["stateId"] == st_adj["stateId"]
            prop = st_ex["count"] / total_obs
            est = st_adj["estimate"]
            assert abs(prop - est) < 1e-10, (
                f"Mismatch at step {sd_ex['horizon']}, state {st_ex['stateId']}: "
                f"excluded prop {prop} vs adjusted est {est}"
            )


def test_r_survival_parity_reference_values_adjusted():
    # Parity check against independent R survival::survfit() references from polarstate parity suite
    # Complete Panel 1 references:
    # Step 0: [1.0, 0.0]
    # Step 1: [2/3, 1/3]
    # Step 2: [1/3, 2/3]
    chart1 = create_reals_distribution_steps(
        COMPLETE_PANEL_1, censoring_heuristic="adjusted"
    )
    sds1 = chart1.spec["stateDistributions"]

    p0 = [st["estimate"] for st in sds1[0]["states"]]
    p1 = [st["estimate"] for st in sds1[1]["states"]]
    p2 = [st["estimate"] for st in sds1[2]["states"]]

    assert p0 == pytest.approx([1.0, 0.0], abs=1e-10)
    assert p1 == pytest.approx([2 / 3, 1 / 3], abs=1e-10)
    assert p2 == pytest.approx([1 / 3, 2 / 3], abs=1e-10)

    # String Panel references (Fixture 3 from polarstate parity tests):
    # States: ["A", "B", "C"]
    # Step 0: [0.4, 0.4, 0.2]
    # Step 1: [0.4, 0.4, 0.2]
    # Step 2: [0.2, 0.2, 0.6]
    chart3 = create_reals_distribution_steps(
        STRING_PANEL, state_order=["A", "B", "C"], censoring_heuristic="adjusted"
    )
    sds3 = chart3.spec["stateDistributions"]

    p0_s3 = [st["estimate"] for st in sds3[0]["states"]]
    p1_s3 = [st["estimate"] for st in sds3[1]["states"]]
    p2_s3 = [st["estimate"] for st in sds3[2]["states"]]

    assert p0_s3 == pytest.approx([0.4, 0.4, 0.2], abs=1e-10)
    assert p1_s3 == pytest.approx([0.4, 0.4, 0.2], abs=1e-10)
    assert p2_s3 == pytest.approx([0.2, 0.2, 0.6], abs=1e-10)


def test_censoring_and_dropout_scenarios():
    # Test Fixture 7 (Independent Terminal Dropout)
    # Step 0: N=6 (3 in state 1, 3 in state 2) -> initial p0 = [0.5, 0.5]
    # Step 1: N=5 (3 in state 1, 2 in state 2)
    # Step 2: N=3 (1 in state 1, 2 in state 2)
    # R survival reference pstate at step 1: [7/12, 5/12] (~0.5833, ~0.4167)
    # R survival reference pstate at step 2: [5/24, 19/24] (~0.2083, ~0.7917)
    chart_adj = create_reals_distribution_steps(
        DROPOUT_PANEL_7, censoring_heuristic="adjusted"
    )
    sds_adj = chart_adj.spec["stateDistributions"]

    p0_est = [st["estimate"] for st in sds_adj[0]["states"]]
    p1_est = [st["estimate"] for st in sds_adj[1]["states"]]
    p2_est = [st["estimate"] for st in sds_adj[2]["states"]]

    assert p0_est == pytest.approx([0.5, 0.5], abs=1e-10)
    assert p1_est == pytest.approx([7 / 12, 5 / 12], abs=1e-10)
    assert p2_est == pytest.approx([5 / 24, 19 / 24], abs=1e-10)

    # Excluded approach uses only available observations at each step
    chart_ex = create_reals_distribution_steps(
        DROPOUT_PANEL_7, censoring_heuristic="excluded"
    )
    sds_ex = chart_ex.spec["stateDistributions"]

    c0 = [st["count"] for st in sds_ex[0]["states"]]
    c1 = [st["count"] for st in sds_ex[1]["states"]]
    c2 = [st["count"] for st in sds_ex[2]["states"]]

    assert c0 == [3, 3]  # total 6
    assert c1 == [3, 2]  # total 5
    assert c2 == [1, 2]  # total 3


def test_baseline_only_subjects():
    # Fixture 9: Subject A only observed at step 0 (state 1).
    # Baseline subjects: A (1), B (1), C (2) -> initial p0 = [2/3, 1/3].
    # At step 1, A drops out; B transitions from 1 to 2; C stays in state 2.
    # AJ occupancy at step 1: state 1 -> 0.0, state 2 -> 1.0.
    chart_adj = create_reals_distribution_steps(
        BASELINE_ONLY_PANEL_9, censoring_heuristic="adjusted"
    )
    sds_adj = chart_adj.spec["stateDistributions"]

    p0_est = [st["estimate"] for st in sds_adj[0]["states"]]
    p1_est = [st["estimate"] for st in sds_adj[1]["states"]]

    assert p0_est == pytest.approx([2 / 3, 1 / 3], abs=1e-10)
    assert p1_est == pytest.approx([0.0, 1.0], abs=1e-10)

    # Excluded approach at step 1 uses only subjects B and C (N=2)
    chart_ex = create_reals_distribution_steps(
        BASELINE_ONLY_PANEL_9, censoring_heuristic="excluded"
    )
    sds_ex = chart_ex.spec["stateDistributions"]

    c0 = [st["count"] for st in sds_ex[0]["states"]]
    c1 = [st["count"] for st in sds_ex[1]["states"]]

    assert c0 == [2, 1]  # total 3
    assert c1 == [0, 2]  # total 2


def test_zero_transition_error_propagation():
    # If all subjects in a positive-occupancy state drop out before step 1 (0 available transitions),
    # polarstate raises a ValueError.
    df_no_trans = pl.DataFrame(
        {
            "id": [1, 2, 2],
            "step": [0, 0, 1],
            "reals": [1, 2, 2],
        }
    )
    # Subject 1 in state 1 at step 0 drops out before step 1 -> 0 transitions from state 1.
    with pytest.raises(ValueError):
        create_reals_distribution_steps(df_no_trans, censoring_heuristic="adjusted")


def test_state_ordering_precedence():
    # 1. Explicit state_order
    chart_ord = create_reals_distribution_steps(COMPLETE_PANEL_1, state_order=[2, 1])
    state_ids = [
        st["stateId"] for st in chart_ord.spec["stateDistributions"][0]["states"]
    ]
    assert state_ids == ["2", "1"]

    # 2. Polars Enum order
    df_enum = pl.DataFrame(
        {
            "id": [1, 1, 2, 2],
            "step": [0, 1, 0, 1],
            "reals": pl.Series(
                ["low", "high", "low", "medium"],
                dtype=pl.Enum(["low", "medium", "high"]),
            ),
        }
    )
    chart_enum = create_reals_distribution_steps(df_enum)
    enum_state_ids = [
        st["stateId"] for st in chart_enum.spec["stateDistributions"][0]["states"]
    ]
    assert enum_state_ids == ["low", "medium", "high"]

    # 3. Numeric deterministic order (10 > 2)
    df_num = pl.DataFrame(
        {
            "id": [1, 1, 2, 2],
            "step": [0, 1, 0, 1],
            "reals": [10, 2, 10, 1],
        }
    )
    chart_num = create_reals_distribution_steps(df_num)
    num_state_ids = [
        st["stateId"] for st in chart_num.spec["stateDistributions"][0]["states"]
    ]
    assert num_state_ids == ["1", "2", "10"]

    # 4. String deterministic lexicographic order
    df_str = pl.DataFrame(
        {
            "id": [1, 1, 2, 2],
            "step": [0, 1, 0, 1],
            "reals": ["zebra", "apple", "zebra", "apple"],
        }
    )
    chart_str = create_reals_distribution_steps(df_str)
    str_state_ids = [
        st["stateId"] for st in chart_str.spec["stateDistributions"][0]["states"]
    ]
    assert str_state_ids == ["apple", "zebra"]


def test_state_identity_collision_prevention():
    # Distinct raw states that map to identical string identifiers must raise ValueError
    df_collision = pl.DataFrame(
        {
            "id": [1, 2],
            "step": [0, 0],
            "reals": pl.Series([1, "1"], dtype=pl.Object),
        }
    )
    with pytest.raises(ValueError, match="State identity collision"):
        create_reals_distribution_steps(df_collision)


def test_state_scale_resolution():
    chart_auto = create_reals_distribution_steps(COMPLETE_PANEL_1, state_scale="auto")
    assert chart_auto.spec["schemaVersion"] == "2.0"

    chart_nom = create_reals_distribution_steps(STRING_PANEL, state_scale="nominal")
    assert chart_nom.spec["schemaVersion"] == "2.0"

    chart_ord = create_reals_distribution_steps(STRING_PANEL, state_scale="ordinal")
    assert chart_ord.spec["schemaVersion"] == "2.0"


def test_input_validations_and_errors():
    # Non-Polars DataFrame input
    with pytest.raises(TypeError, match="Polars DataFrame"):
        create_reals_distribution_steps([1, 2, 3])  # type: ignore[arg-type]

    # Empty DataFrame
    with pytest.raises(ValueError, match="cannot be empty"):
        create_reals_distribution_steps(pl.DataFrame())

    # Missing required columns
    with pytest.raises(ValueError, match="Missing required column"):
        create_reals_distribution_steps(pl.DataFrame({"id": [1], "step": [0]}))

    # Nulls in ID / step / state
    df_null_id = pl.DataFrame({"id": [None, 1], "step": [0, 1], "reals": [1, 2]})
    with pytest.raises(ValueError, match="contains null"):
        create_reals_distribution_steps(df_null_id)

    df_null_step = pl.DataFrame({"id": [1, 2], "step": [0, None], "reals": [1, 2]})
    with pytest.raises(ValueError, match="contains null"):
        create_reals_distribution_steps(df_null_step)

    df_null_state = pl.DataFrame({"id": [1, 2], "step": [0, 1], "reals": [1, None]})
    with pytest.raises(ValueError, match="contains null"):
        create_reals_distribution_steps(df_null_state)

    # Non-integer step
    df_float_step = pl.DataFrame({"id": [1, 2], "step": [0.5, 1.2], "reals": [1, 2]})
    with pytest.raises(ValueError, match="integer"):
        create_reals_distribution_steps(df_float_step)

    # Negative step
    df_neg_step = pl.DataFrame({"id": [1, 2], "step": [-1, 0], "reals": [1, 2]})
    with pytest.raises(ValueError, match="non-negative"):
        create_reals_distribution_steps(df_neg_step)

    # Duplicate subject-step pairs
    df_dup = pl.DataFrame({"id": [1, 1], "step": [0, 0], "reals": [1, 2]})
    with pytest.raises(ValueError, match="Duplicate records"):
        create_reals_distribution_steps(df_dup)

    # Invalid state_order
    with pytest.raises(
        ValueError, match="state_order must contain every observed state"
    ):
        create_reals_distribution_steps(
            COMPLETE_PANEL_1, state_order=[1]
        )  # missing state 2

    # Unsupported arguments
    with pytest.raises(ValueError, match="Unsupported renderer"):
        create_reals_distribution_steps(COMPLETE_PANEL_1, renderer="invalid")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="Unsupported censoring_heuristic"):
        create_reals_distribution_steps(COMPLETE_PANEL_1, censoring_heuristic="invalid")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="Unsupported state_scale"):
        create_reals_distribution_steps(COMPLETE_PANEL_1, state_scale="invalid")  # type: ignore[arg-type]


def test_behavioral_invariants():
    # Input row order invariance
    df_shuffled = COMPLETE_PANEL_1.sample(fraction=1.0, shuffle=True, seed=42)
    spec_orig = create_reals_distribution_steps(COMPLETE_PANEL_1).spec
    spec_shuffled = create_reals_distribution_steps(df_shuffled).spec
    assert spec_orig == spec_shuffled

    # Single step data
    df_single = pl.DataFrame(
        {"id": [1, 2, 3], "step": [0, 0, 0], "reals": ["A", "B", "A"]}
    )
    chart_single = create_reals_distribution_steps(df_single)
    assert len(chart_single.spec["stateDistributions"]) == 1
    assert chart_single.spec["stateDistributions"][0]["horizon"] == 0

    # Non-zero starting step
    df_nonzero_start = pl.DataFrame(
        {"id": [1, 1, 2, 2], "step": [5, 6, 5, 6], "reals": [1, 2, 1, 1]}
    )
    chart_nz = create_reals_distribution_steps(df_nonzero_start)
    horizons = [sd["horizon"] for sd in chart_nz.spec["stateDistributions"]]
    assert horizons == [5, 6]

    # Probabilities sum to 1.0 (adjusted AJ)
    chart_adj = create_reals_distribution_steps(
        COMPLETE_PANEL_2, censoring_heuristic="adjusted"
    )
    for sd in chart_adj.spec["stateDistributions"]:
        prob_sum = sum(st["estimate"] for st in sd["states"])
        assert abs(prob_sum - 1.0) < 1e-10

    # Counts sum to step total (excluded)
    chart_ex = create_reals_distribution_steps(
        DROPOUT_PANEL_7, censoring_heuristic="excluded"
    )
    sds_ex = chart_ex.spec["stateDistributions"]
    assert sum(st["count"] for st in sds_ex[0]["states"]) == 6
    assert sum(st["count"] for st in sds_ex[1]["states"]) == 5
    assert sum(st["count"] for st in sds_ex[2]["states"]) == 3

    # JSON serializability
    spec_json = json.dumps(chart_adj.spec)
    assert isinstance(spec_json, str)
    deserialized = json.loads(spec_json)
    assert deserialized["schemaVersion"] == "2.0"

    # Time-based producer unchanged
    chart_times = create_reals_distribution_times([1, 0], [10.0, 20.0], [15.0])
    assert chart_times.spec["schemaVersion"] == "2.0"

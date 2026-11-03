"""R/Python parity checks for continuous observed longitudinal outcomes."""

import math

import polars as pl
import pytest

from rtichoke import create_reals_distribution_steps

# Exact values from rtichoke/data-raw/continuous_outcomes_steps.R.
R_ROWS = [
    (1, 0, 10), (1, 1, 15), (1, 3, 20),
    (2, 0, 12), (2, 1, 17), (2, 3, 25),
    (3, 0, 14), (3, 1, None), (3, 3, 30),
    (4, 0, 16), (4, 1, 20), (4, 3, None),
    (5, 0, 18), (5, 1, 22), (5, 3, 35),
    (6, 0, 20), (6, 1, 24),
]


def fixture() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "id": [row[0] for row in R_ROWS],
            "step": [row[1] for row in R_ROWS],
            "reals": pl.Series([row[2] for row in R_ROWS], dtype=pl.Float64),
        }
    )


def spec(data=None, bins=(10, 20, 30, 40)):
    return create_reals_distribution_steps(
        fixture() if data is None else data,
        reals_scale="continuous",
        bins=bins,
    ).spec


def test_exact_r_fixture_mass_and_missingness():
    result = spec()
    assert result["schemaVersion"] == "2.0"
    assert result["type"] == "continuous_outcome_distribution"
    assert result["xAxisKind"] == "step"
    assert result["estimator"] == "raw"
    assert result["binEdges"] == [10.0, 20.0, 30.0, 40.0]
    assert result["evaluations"] == [{"id": "evaluation-1", "population": "population"}]
    distributions = result["distributions"]
    assert [item["step"] for item in distributions] == [0, 1, 3]
    assert [item["referenceSampleSize"] for item in distributions] == [6, 6, 6]
    assert [item["observedCount"] for item in distributions] == [6, 5, 4]
    assert [item["unobservedCount"] for item in distributions] == [0, 1, 2]
    assert [item["binCounts"] for item in distributions] == [
        [5, 1, 0], [2, 3, 0], [0, 3, 1]
    ]
    for item in distributions:
        assert sum(item["binCounts"]) == item["observedCount"]


def test_boundary_assignment_and_equal_width_bins():
    df = pl.DataFrame({"id": [1, 2, 3, 4], "step": [2] * 4, "reals": [0., 1., 2., 3.]})
    result = spec(df, [0, 1, 2, 3])
    assert result["distributions"][0]["binCounts"] == [1, 1, 2]
    assert spec(df, 3)["binEdges"] == [0.0, 1.0, 2.0, 3.0]


def test_constant_values_and_zero_observed_supplied_step():
    df = pl.DataFrame({"id": [1, 1, 2, 2], "step": [0, 1, 0, 1], "reals": [7., None, 7., None]})
    result = spec(df, 2)
    assert result["binEdges"] == [6.5, 7.0, 7.5]
    assert result["distributions"][0]["binCounts"] == [0, 2]
    assert result["distributions"][1]["binCounts"] == [0, 0]
    assert result["distributions"][1]["unobservedCount"] == 2


@pytest.mark.parametrize(
    ("rows", "bins", "error"),
    [
        ([(1, 0, 1.), (1, 0, 2.)], 3, "unique"),
        ([(None, 0, 1.)], 3, "missing values"),
        ([(1, None, 1.)], 3, "missing values"),
        ([(1, -1, 1.)], 3, "non-negative"),
        ([(1, 1.5, 1.)], 3, "integer"),
        ([(1, 0, float("inf"))], 3, "infinite"),
        ([(1, 0, 1.)], 0, "positive integer"),
        ([(1, 0, 1.)], [1, 1, 2], "strictly increasing"),
        ([(1, 0, 1.)], [2, 3], "outside"),
    ],
)
def test_invalid_data_and_bins(rows, bins, error):
    df = pl.DataFrame(rows, schema=["id", "step", "reals"], orient="row")
    with pytest.raises(ValueError, match=error):
        spec(df, bins)


def test_all_missing_values_and_missing_column():
    df = pl.DataFrame({"id": [1, 2], "step": [0, 0], "reals": pl.Series([None, None], dtype=pl.Float64)})
    with pytest.raises(ValueError, match="All outcomes"):
        spec(df)
    with pytest.raises(ValueError, match="Missing required columns"):
        spec(df.drop("reals"))


def test_continuous_rejects_order_and_adjustment():
    with pytest.raises(ValueError, match="reals_order"):
        create_reals_distribution_steps(fixture(), reals_scale="continuous", reals_order=[1, 2])
    with pytest.raises(ValueError, match="Adjusted censoring"):
        create_reals_distribution_steps(fixture(), reals_scale="continuous", censoring_heuristic="adjusted")


def test_numeric_auto_remains_ordinal():
    result = create_reals_distribution_steps(fixture())
    assert result.spec["type"] == "outcome_distribution"
    assert result.spec["stateScale"] == "ordinal"


def test_renderer_exports_continuous(tmp_path):
    chart = create_reals_distribution_steps(fixture(), reals_scale="continuous", bins=[10, 20, 30, 40])
    output = chart.write_html(tmp_path / "chart.html")
    html = output.read_text()
    assert "renderContinuousOutcomeDistribution" in html
    assert "continuous_outcome_distribution" in html
    assert (tmp_path / "rtichoke-viz.js").exists()

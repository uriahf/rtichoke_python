"""Compare polarstate and the browser adapter with independently fitted R AJ."""

import json
import os
import shutil
import subprocess
from pathlib import Path

import polars as pl
import polarstate
import pytest

from rtichoke import create_reals_distribution_steps

FIXTURES = Path(__file__).parent / "fixtures" / "multistate_aj"


@pytest.mark.parametrize("case", ["complete", "right_censored"])
def test_polarstate_against_r_survfit(case):
    reference = json.loads((FIXTURES / "r_multistate_reference.json").read_text())
    scenario = reference["cases"][case]
    panel = pl.DataFrame(scenario["panel"]).rename({"state": "reals"})
    transitions = polarstate.prepare_transition_table_steps(
        panel,
        id_col="id",
        step_col="step",
        state_col="reals",
        state_order=reference["states"],
    )
    predictions = polarstate.predict_aj_estimates(transitions)
    chart = create_reals_distribution_steps(
        panel, reals_order=reference["states"], censoring_heuristic="adjusted"
    )
    assert [d["horizon"] for d in chart.spec["stateDistributions"]] == reference[
        "steps"
    ]
    for distribution in chart.spec["stateDistributions"]:
        step = distribution["horizon"]
        expected = scenario["occupancy"][str(step)]
        actual = predictions.filter(pl.col("step") == step)
        assert actual.height == len(expected)
        for row in actual.iter_rows(named=True):
            assert row["state_occupancy_probability"] == pytest.approx(
                expected[str(row["state"])], abs=1e-12
            )
        assert {s["stateId"]: s["estimate"] for s in distribution["states"]} == (
            pytest.approx(expected, abs=1e-12)
        )


def test_r_reference_regeneration(tmp_path):
    rscript = os.environ.get("RSCRIPT") or shutil.which("Rscript")
    if rscript is None:
        raise pytest.skip.Exception(
            "Set RSCRIPT or install Rscript to regenerate the R reference"
        )
    output = tmp_path / "reference.json"
    subprocess.run(
        [rscript, str(FIXTURES / "generate_r_multistate_reference.R"), str(output)],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    regenerated = json.loads(output.read_text())
    committed = json.loads((FIXTURES / "r_multistate_reference.json").read_text())
    assert regenerated["reference_package"] == committed["reference_package"]
    assert regenerated["method"] == committed["method"]
    assert regenerated["steps"] == committed["steps"]
    assert regenerated["states"] == committed["states"]
    for case in committed["cases"]:
        assert regenerated["cases"][case]["panel"] == committed["cases"][case]["panel"]
        assert (
            regenerated["cases"][case]["intervals"]
            == committed["cases"][case]["intervals"]
        )
        for step in committed["cases"][case]["occupancy"]:
            assert regenerated["cases"][case]["occupancy"][step] == pytest.approx(
                committed["cases"][case]["occupancy"][step], abs=1e-12
            )

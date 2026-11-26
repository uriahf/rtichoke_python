"""Compare Python report calibration to archived R PBC overlay points."""

import csv
import json
import os
import re
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest

from rtichoke import create_summary_report_times

FIXTURES = Path(__file__).parent / "fixtures"
MODELS = ("Age only", "Age + labs")
HORIZONS = (1.0, 3.0, 5.0)


def test_pbc_discrete_calibration_matches_r(tmp_path):
    rscript = os.environ.get("RSCRIPT") or shutil.which("Rscript")
    if rscript is None:
        pytest.skip(
            "Set RSCRIPT or install R with survival to generate PBC predictions"
        )

    inputs = tmp_path / "predictions.csv"
    subprocess.run(
        [rscript, str(FIXTURES / "generate_pbc_predictions.R"), str(inputs)],
        check=True,
    )
    with inputs.open(newline="") as stream:
        subjects = list(csv.DictReader(stream))
    assert len(subjects) == 168
    predictions = {
        h: {
            model: np.array([float(row[f"{i}_{int(h)}"]) for row in subjects])
            for i, model in enumerate(MODELS, start=1)
        }
        for h in HORIZONS
    }
    report = create_summary_report_times(
        predictions[5.0],
        np.array([int(row["event"]) for row in subjects]),
        np.array([float(row["time"]) for row in subjects]),
        list(HORIZONS),
        calibration_probs_by_horizon=predictions,
        binning="type7",
        n_bins=5,
        by=0.2,
        output_file=tmp_path / "pbc.html",
    )
    match = re.search(
        r'<script id="rtichoke-report-spec" type="application/json">(.*?)</script>',
        report.read_text(encoding="utf-8"),
        re.S,
    )
    assert match is not None
    sections = json.loads(match[1])["sections"]
    section = next(section for section in sections if section["id"] == "calibration")
    spec = next(
        item["spec"] for item in section["items"] if item["id"] == "calibration"
    )
    series = {
        row["id"]: (row["horizon"], row["display"]["label"]) for row in spec["series"]
    }
    actual = defaultdict(list)
    for point in spec["data"]:
        actual[series[point["seriesId"]]].append(point)

    expected = defaultdict(list)
    with (FIXTURES / "pbc_r_discrete_calibration.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            expected[(float(row["horizon_years"]), row["model"])].append(row)

    assert actual.keys() == expected.keys()
    assert len(expected) == 6
    for key, points in actual.items():
        reference = sorted(expected[key], key=lambda row: int(row["bin"]))
        assert len(points) == len(reference) == 5
        for point, row in zip(
            sorted(points, key=lambda p: p["predicted"]), reference, strict=True
        ):
            assert point["total"] == int(row["total"]), key
            for field in ("predicted", "observed"):
                assert point[field] == pytest.approx(
                    float(row[field]), abs=1e-10, rel=0
                ), key

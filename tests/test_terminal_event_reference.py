"""Independent terminal-event KM/AJ references; no report/renderer dependencies."""

import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from polarstate import predict_aj_estimates, prepare_event_table

from rtichoke.performance_data.performance_data_times import (
    _compute_population_event_risk_times,
)
from rtichoke.processing.adjustments import _extract_excluded_events

FIXTURES = Path(__file__).parent / "fixtures" / "terminal_event"
REFERENCE = json.loads((FIXTURES / "reference.json").read_text())
CASES = [
    (name, key, row)
    for name, case in REFERENCE["cases"].items()
    for key, rows in case["variants"].items()
    for row in rows
]


def parameters(cases):
    return [
        pytest.param(name, key, row, id=f"{name}-{key}-{row['horizon']}")
        for name, key, row in cases
    ]


ROWS = parameters(CASES)


def ids(value):
    """Normalize jsonlite's auto-unboxed singleton character vectors."""
    return [value] if isinstance(value, str) else value


@pytest.mark.parametrize("name,key,row", ROWS)
def test_retention_and_complete_reference_distributions(name, key, row):
    cohort = REFERENCE["cases"][name]["data"]
    censoring, competing = key.split(":")
    h = row["horizon"]
    excluded_c = [
        s["id"]
        for s in cohort
        if censoring == "excluded" and s["event"] == 0 and s["time"] < h
    ]
    excluded_k = [
        s["id"]
        for s in cohort
        if competing == "excluded" and s["event"] == 2 and s["time"] <= h
    ]
    retained = [s["id"] for s in cohort if s["id"] not in excluded_c + excluded_k]
    assert ids(row["retained_ids"]) == retained
    assert ids(row["censored_excluded_ids"]) == excluded_c
    assert ids(row["competing_excluded_ids"]) == excluded_k
    assert row["retained_n"] == len(retained)
    assert row["original_n"] == len(cohort)
    assert row["censored_excluded_count"] == len(excluded_c)
    assert row["competing_excluded_count"] == len(excluded_k)
    probabilities = row["probabilities"]
    expected_states = (
        {"primary", "competing", "event_free"}
        if competing == "adjusted_as_negative"
        else {
            "composite" if competing == "adjusted_as_composite" else "primary",
            "event_free",
        }
    )
    assert set(probabilities) == expected_states
    if not retained:
        assert row["status"] == "undefined_empty_population"
        assert all(p is None for p in probabilities.values())
        return
    assert all(0 <= p <= 1 for p in probabilities.values())
    assert sum(probabilities.values()) == pytest.approx(1, abs=1e-12)
    if row["cuminc"] is not None:
        assert row["cuminc"] == pytest.approx(
            {s: probabilities[s] for s in ("primary", "competing")}, abs=1e-12
        )


@pytest.mark.parametrize(
    "key,expected",
    [
        ("excluded:excluded", (1 / 2, 1 / 2)),
        ("excluded:adjusted_as_censored", (3 / 8, 5 / 8)),
        ("excluded:adjusted_as_negative", (1 / 3, 1 / 3, 1 / 3)),
        ("excluded:adjusted_as_composite", (2 / 3, 1 / 3)),
        ("adjusted:excluded", (4 / 9, 5 / 9)),
        ("adjusted:adjusted_as_censored", (11 / 32, 21 / 32)),
        ("adjusted:adjusted_as_negative", (5 / 16, 5 / 16, 3 / 8)),
        ("adjusted:adjusted_as_composite", (5 / 8, 3 / 8)),
    ],
)
def test_fixture_a_all_eight_full_vectors(key, expected):
    rows = REFERENCE["cases"]["fixture_a"]["variants"][key]
    row = next(r for r in rows if r["horizon"] == 10)
    assert list(row["probabilities"].values()) == pytest.approx(expected, abs=1e-12)
    h5 = next(r for r in rows if r["horizon"] == 5)
    mode = key.split(":")[1]
    expected5 = {
        "excluded": (1 / 7, 6 / 7),
        "adjusted_as_censored": (1 / 8, 7 / 8),
        "adjusted_as_negative": (1 / 8, 1 / 8, 3 / 4),
        "adjusted_as_composite": (1 / 4, 3 / 4),
    }[mode]
    assert list(h5["probabilities"].values()) == pytest.approx(expected5, abs=1e-12)
    assert ids(h5["retained_ids"]) == list(
        "ACDEFGH" if mode == "excluded" else "ABCDEFGH"
    )
    expected10 = (
        "AFGH"
        if key == "excluded:excluded"
        else "ABEFGH"
        if key.startswith("excluded:")
        else "ACDFGH"
        if mode == "excluded"
        else "ABCDEFGH"
    )
    assert ids(row["retained_ids"]) == list(expected10)


def test_zero_and_undefined_contract():
    variants = REFERENCE["cases"]["zero_ties"]["variants"]
    p = variants["adjusted:adjusted_as_negative"][0]
    assert p["probabilities"] == {"primary": 0.25, "competing": 0.25, "event_free": 0.5}
    assert ids(p["retained_ids"]) == list("ABCD")
    p = variants["excluded:excluded"][0]
    assert ids(p["retained_ids"]) == list("ACD")
    assert p["probabilities"]["primary"] == pytest.approx(1 / 3)
    rows = REFERENCE["cases"]["all_censored"]["variants"]
    assert rows["excluded:excluded"][-1]["status"] == "undefined_empty_population"
    assert rows["adjusted:excluded"][-1]["status"] == "beyond_support_plateau"


@pytest.mark.parametrize(
    "name,key,row", parameters([p for p in CASES if p[2]["retained_n"] == 0])
)
@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="Known discrepancy: empty retained population returns finite zero risk; independent contract is undefined. See fixtures/terminal_event/README.md.",
)
def test_empty_population_must_not_be_reported_as_zero_risk(name, key, row):
    cohort = REFERENCE["cases"][name]["data"]
    c, k = key.split(":")
    actual = _compute_population_event_risk_times(
        np.array([s["event"] for s in cohort]),
        np.array([s["time"] for s in cohort], dtype=float),
        row["horizon"],
        c,
        k,
    )
    assert actual is None or not np.isfinite(actual)


@pytest.mark.parametrize(
    "name,key,row", parameters([p for p in CASES if p[2]["retained_n"] > 0])
)
def test_existing_population_risk_and_exclusion_producers(name, key, row):
    cohort = REFERENCE["cases"][name]["data"]
    c, k = key.split(":")
    reals = np.array([s["event"] for s in cohort])
    times = np.array([s["time"] for s in cohort], dtype=float)
    actual = _compute_population_event_risk_times(reals, times, row["horizon"], c, k)
    state = "composite" if k == "adjusted_as_composite" else "primary"
    assert actual == pytest.approx(row["probabilities"][state], abs=1e-12)
    counts = _extract_excluded_events(
        prepare_event_table(pl.DataFrame({"reals": reals, "times": times})),
        [row["horizon"]],
        c,
        k,
    ).row(0, named=True)
    assert counts["real_censored_est"] == row["censored_excluded_count"]
    assert counts["real_competing_est"] == row["competing_excluded_count"]


@pytest.mark.parametrize(
    "name,key,row", parameters([p for p in CASES if p[2]["retained_n"] > 0])
)
def test_polarstate_complete_vectors_against_independent_r(name, key, row):
    _, k = key.split(":")
    cohort = REFERENCE["cases"][name]["data"]
    retained = [s for s in cohort if s["id"] in ids(row["retained_ids"])]
    events = [s["event"] for s in retained]
    if k == "adjusted_as_censored":
        events = [0 if e == 2 else e for e in events]
    if k == "adjusted_as_composite":
        events = [1 if e == 2 else e for e in events]
    estimates = predict_aj_estimates(
        prepare_event_table(
            pl.DataFrame(
                {"reals": events, "times": [float(s["time"]) for s in retained]}
            )
        ),
        pl.Series([float(row["horizon"])]),
        full_event_table=False,
    ).row(0, named=True)
    expected = row["probabilities"]
    assert estimates["state_occupancy_probability_0"] == pytest.approx(
        expected["event_free"], abs=1e-12
    )
    assert estimates["state_occupancy_probability_1"] == pytest.approx(
        expected.get("primary", expected.get("composite")), abs=1e-12
    )
    if k == "adjusted_as_negative":
        assert estimates["state_occupancy_probability_2"] == pytest.approx(
            expected["competing"], abs=1e-12
        )


def test_independent_r_regeneration(tmp_path):
    rscript = os.environ.get("RSCRIPT") or shutil.which("Rscript")
    if rscript is None:
        raise pytest.skip.Exception(
            "Set RSCRIPT to regenerate survival/cmprsk references"
        )
    output = tmp_path / "reference.json"
    subprocess.run(
        [rscript, str(FIXTURES / "generate_reference.R"), str(output)],
        check=True,
        capture_output=True,
        text=True,
    )
    regenerated = json.loads(output.read_text())
    assert set(regenerated["versions"]) == {"R", "survival", "cmprsk", "jsonlite"}
    assert regenerated["contract"] == REFERENCE["contract"]
    # Versions are recorded, not pinned: independently verify numerical outputs on CI's R release.
    for name, case in REFERENCE["cases"].items():
        assert regenerated["cases"][name]["data"] == case["data"]
        for key, rows in case["variants"].items():
            for actual, expected in zip(
                regenerated["cases"][name]["variants"][key], rows, strict=True
            ):
                for field in expected:
                    if (
                        field in ("probabilities", "cuminc")
                        and expected[field] is not None
                    ):
                        assert actual[field] == pytest.approx(
                            expected[field], abs=1e-12
                        )
                    else:
                        assert actual[field] == expected[field]

from pathlib import Path
from typing import Any, cast

import numpy as np

from rtichoke._viz_browser import (
    _calibration_spec_from_curve_list,
    _roc_spec_from_performance_data,
    _write_calibration_browser_html,
    _write_roc_browser_html,
)
from rtichoke.calibration.calibration import _create_calibration_curve_list
from rtichoke.performance_data.performance_data import prepare_performance_data


def _real_roc_performance_data():
    return prepare_performance_data(
        probs={"Model A": np.array([0.05, 0.10, 0.20, 0.35, 0.55, 0.70, 0.85, 0.95])},
        reals=np.array([0, 0, 0, 1, 0, 1, 1, 1]),
        by=0.1,
    )


def _real_calibration_curve_list():
    return _create_calibration_curve_list(
        probs={
            "Model A": np.array(
                [0.03, 0.08, 0.12, 0.18, 0.25, 0.32, 0.40, 0.50, 0.62, 0.75, 0.88, 0.96]
            )
        },
        reals=np.array([0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1]),
    )


def test_real_roc_output_maps_to_canonical_spec():
    performance_data = _real_roc_performance_data()

    spec = _roc_spec_from_performance_data(performance_data)
    spec = cast(dict[str, Any], spec)

    assert spec["schemaVersion"] == "1.0"
    assert spec["type"] == "roc"
    assert spec["references"] == [{"type": "identity"}]
    assert spec["data"]
    assert {row["model"] for row in spec["data"]} == {"Model A"}
    assert all(0 <= row["sensitivity"] <= 1 for row in spec["data"])
    assert all(0 <= row["specificity"] <= 1 for row in spec["data"])


def test_real_calibration_output_maps_to_canonical_spec():
    spec = _calibration_spec_from_curve_list(_real_calibration_curve_list())
    spec = cast(dict[str, Any], spec)

    assert spec["schemaVersion"] == "1.0"
    assert spec["type"] == "calibration"
    assert spec["references"] == [{"type": "identity"}]
    assert spec["data"]
    assert {row["model"] for row in spec["data"]} == {"Model A"}
    assert all(row["method"] == "discrete" for row in spec["data"])
    assert all(0 <= row["predicted"] <= 1 for row in spec["data"])
    assert all(0 <= row["observed"] <= 1 for row in spec["data"])
    assert spec["distribution"]


def test_roc_browser_proof_uses_vendored_assets(tmp_path: Path):
    output = _write_roc_browser_html(
        _real_roc_performance_data(),
        tmp_path / "index.html",
    )

    html = output.read_text(encoding="utf-8")
    assert 'import { renderRoc } from "./rtichoke-viz.js"' in html
    assert '"type": "roc"' in html
    assert (tmp_path / "rtichoke-viz.js").stat().st_size > 0
    assert (tmp_path / "rtichoke-viz.css").stat().st_size > 0


def test_calibration_browser_proof_uses_vendored_assets(tmp_path: Path):
    output = _write_calibration_browser_html(
        _real_calibration_curve_list(),
        tmp_path / "index.html",
    )

    html = output.read_text(encoding="utf-8")
    assert 'import { renderCalibration } from "./rtichoke-viz.js"' in html
    assert '"type": "calibration"' in html
    assert (tmp_path / "rtichoke-viz.js").stat().st_size > 0
    assert (tmp_path / "rtichoke-viz.css").stat().st_size > 0

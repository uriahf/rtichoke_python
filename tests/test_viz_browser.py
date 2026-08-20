from pathlib import Path

import numpy as np

from rtichoke._viz_browser import (
    _roc_spec_from_performance_data,
    _write_roc_browser_html,
)
from rtichoke.performance_data.performance_data import prepare_performance_data


def _real_roc_performance_data():
    return prepare_performance_data(
        probs={
            "Model A": np.array(
                [0.05, 0.10, 0.20, 0.35, 0.55, 0.70, 0.85, 0.95]
            )
        },
        reals=np.array([0, 0, 0, 1, 0, 1, 1, 1]),
        by=0.1,
    )


def test_real_roc_output_maps_to_canonical_spec():
    performance_data = _real_roc_performance_data()

    spec = _roc_spec_from_performance_data(performance_data)

    assert spec["schemaVersion"] == "1.0"
    assert spec["type"] == "roc"
    assert spec["references"] == [{"type": "identity"}]
    assert spec["data"]
    assert {row["model"] for row in spec["data"]} == {"Model A"}
    assert all(0 <= row["sensitivity"] <= 1 for row in spec["data"])
    assert all(0 <= row["specificity"] <= 1 for row in spec["data"])


def test_browser_proof_uses_vendored_assets(tmp_path: Path):
    output = _write_roc_browser_html(
        _real_roc_performance_data(),
        tmp_path / "index.html",
    )

    html = output.read_text(encoding="utf-8")
    assert 'import { renderRoc } from "./rtichoke-viz.js"' in html
    assert '"type": "roc"' in html
    assert (tmp_path / "rtichoke-viz.js").stat().st_size > 0
    assert (tmp_path / "rtichoke-viz.css").stat().st_size > 0

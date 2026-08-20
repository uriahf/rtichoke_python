import numpy as np

from rtichoke.summary_report.summary_report_v2 import create_summary_report


def test_shared_outcomes_render_one_prevalence_population(tmp_path):
    probs = {
        "Model A": np.array([0.1, 0.3, 0.6, 0.8]),
        "Model B": np.array([0.2, 0.4, 0.5, 0.7]),
    }
    reals = np.array([0, 0, 1, 1])
    output = tmp_path / "report.html"

    create_summary_report(probs, reals, output_file=output, by=0.1)

    html = output.read_text(encoding="utf-8")
    assert "const prevalenceRows=SUM.slice(0,1);" in html
    assert "prevalenceRows.forEach" in html


def test_summary_report_uses_r_style_auc_widget(tmp_path):
    probs = {
        "Model A": np.array([0.1, 0.3, 0.6, 0.8]),
        "Model B": np.array([0.2, 0.4, 0.5, 0.7]),
    }
    reals = np.array([0, 0, 1, 1])
    output = tmp_path / "report.html"

    create_summary_report(probs, reals, output_file=output, by=0.1)

    html = output.read_text(encoding="utf-8")
    assert 'class="summary-table auc-table"' in html
    assert "<th>AUROC</th>" in html
    assert "value.toFixed(2)" in html
    assert "background:green" in html
    assert "model-badge" in html


def test_summary_report_uses_paginated_modular_performance_table(tmp_path):
    probs = {"Model A": np.linspace(0.01, 0.99, 20)}
    reals = np.array([0, 1] * 10)
    output = tmp_path / "report.html"

    create_summary_report(probs, reals, output_file=output, by=0.05)

    html = output.read_text(encoding="utf-8")
    assert "const PAGE_SIZE = 10;" in html
    assert "rt-page-info" in html
    assert "filtered.length" in html
    assert "perf(R.tables.threshold,'#table-threshold',false)" not in html


def test_summary_report_has_r_style_performance_filters(tmp_path):
    probs = {
        "Model A": np.array([0.1, 0.3, 0.6, 0.8]),
        "Model B": np.array([0.2, 0.4, 0.5, 0.7]),
    }
    reals = np.array([0, 0, 1, 1])
    output = tmp_path / "report.html"

    create_summary_report(probs, reals, output_file=output, by=0.1)

    html = output.read_text(encoding="utf-8")
    assert 'className="rt-filters"' in html
    assert 'textContent="Model"' in html
    assert '"Probability Threshold"' in html
    assert '"Predicted Positives Condition Rate (PPCR)"' in html
    assert 'className="rt-dual-range"' in html
    assert "selected.has(model)" in html


def test_performance_table_recovers_tp_when_legacy_payload_omits_it(tmp_path):
    probs = {"Model A": np.array([0.1, 0.4, 0.7, 0.9])}
    reals = np.array([0, 1, 0, 1])
    output = tmp_path / "report.html"

    create_summary_report(probs, reals, output_file=output, by=0.1)

    html = output.read_text(encoding="utf-8")
    assert "r.true_positives == null" in html
    assert "num(r.predicted_positives)-fp" in html

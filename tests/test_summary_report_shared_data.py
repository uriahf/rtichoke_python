import numpy as np

import rtichoke.summary_report.summary_report as summary_report


def test_summary_report_prepares_each_stratification_once(monkeypatch, tmp_path):
    original = summary_report.prepare_performance_data
    calls = []

    def counted(*args, **kwargs):
        calls.append(tuple(kwargs.get("stratified_by", ())))
        return original(*args, **kwargs)

    monkeypatch.setattr(summary_report, "prepare_performance_data", counted)
    summary_report.create_summary_report(
        {"model": np.array([0.1, 0.3, 0.6, 0.8])},
        np.array([0, 0, 1, 1]),
        output_file=tmp_path / "report.html",
        by=0.1,
    )

    assert calls.count(("probability_threshold",)) == 1
    assert calls.count(("ppcr",)) == 1
    assert len(calls) == 2

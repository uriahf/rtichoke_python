import numpy as np

from rtichoke.summary_report.summary_report import create_summary_report


def test_create_summary_report_writes_native_html(tmp_path):
    probs = {"model": np.array([0.05, 0.2, 0.4, 0.7, 0.9])}
    reals = np.array([0, 0, 1, 1, 1])
    output = tmp_path / "report.html"

    result = create_summary_report(probs, reals, output_file=output, by=0.1)

    html = output.read_text(encoding="utf-8")
    assert result == output
    assert "Model Performance Summary" in html
    assert "ROC Curve" in html
    assert "Precision-Recall Curve" in html
    assert "Gains Curve" in html
    assert "Lift Curve" in html
    assert "Decision Curve" in html
    assert "const specs=" in html
    assert "renderTabs();" in html
    assert "draw(specs[0]);" in html
    assert "d3.scaleLinear()" in html
    assert "send_requests_to_rtichoke_r" not in html
    assert "quarto" not in html.lower()

from rtichoke.summary_report.calibration_renderer import calibration_renderer_source


def test_calibration_renderer_asset_is_available():
    source = calibration_renderer_source()
    assert "function calibration(type, sel)" in source
    # Keep this test aligned with the canonical 600px Plotly geometry used by
    # the parity renderer rather than the earlier hand-tuned 550px prototype.
    assert "const W = 600, H = 600" in source
    assert "MAIN_TOP = 100, MAIN_BOTTOM = 424" in source
    assert "HIST_TOP = 444, HIST_BOTTOM = 525" in source
    assert 'text("Predicted")' in source
    assert 'text("Observed")' in source
    assert "showTip" in source
    assert "hoverColor" in source
    assert "c.colors[groupOf(d)]" in source


def test_calibration_renderer_is_safe_to_inline_in_report_script():
    source = calibration_renderer_source()
    assert "</script>" not in source.lower()
    assert "calibration('smooth'" not in source
    assert 'calibration("smooth"' not in source

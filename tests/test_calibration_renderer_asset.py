from rtichoke.summary_report.calibration_renderer import calibration_renderer_source


def test_calibration_renderer_asset_is_available():
    source = calibration_renderer_source()
    assert "function calibration(type, sel)" in source
    # Match the geometry used by the actual R summary-report calibration call.
    assert "const W = 550, H = 550" in source
    assert "MAIN_TOP = 55, MAIN_BOTTOM = 409.9" in source
    assert "HIST_TOP = 428.1, HIST_BOTTOM = 510" in source
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

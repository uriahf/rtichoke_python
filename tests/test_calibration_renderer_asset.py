from rtichoke.summary_report.calibration_renderer import calibration_renderer_source


def test_calibration_renderer_asset_is_available():
    source = calibration_renderer_source()
    assert "function calibration(type, sel)" in source
    assert "MAIN_BOTTOM = 409.9" in source
    assert "HIST_TOP = 428.1" in source
    assert 'text("Predicted")' in source
    assert 'text("Observed")' in source
    assert "showTip" in source
    assert "c.colors[group]" in source


def test_calibration_renderer_is_safe_to_inline_in_report_script():
    source = calibration_renderer_source()
    assert "</script>" not in source.lower()
    assert "calibration('smooth'" not in source
    assert "calibration(\"smooth\"" not in source

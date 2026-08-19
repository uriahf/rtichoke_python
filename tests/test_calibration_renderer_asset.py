from rtichoke.summary_report.calibration_renderer import calibration_renderer_source


def test_calibration_renderer_asset_is_available():
    source = calibration_renderer_source()
    assert "function calibration(type, sel)" in source
    assert "MAIN_BOTTOM = 409.9" in source
    assert "HIST_TOP = 428.1" in source
    assert 'text("Predicted")' in source
    assert 'text("Observed")' in source

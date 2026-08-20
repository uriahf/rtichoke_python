import rtichoke


EXPECTED_PUBLIC_EXPORTS = {
    "create_roc_curve",
    "create_roc_curve_times",
    "plot_roc_curve",
    "create_lift_curve",
    "create_lift_curve_times",
    "plot_lift_curve",
    "create_precision_recall_curve",
    "create_precision_recall_curve_times",
    "plot_precision_recall_curve",
    "create_gains_curve",
    "create_gains_curve_times",
    "plot_gains_curve",
    "create_calibration_curve",
    "create_calibration_curve_times",
    "create_decision_curve",
    "create_decision_curve_times",
    "plot_decision_curve",
    "prepare_performance_data",
    "prepare_binned_classification_data",
    "prepare_performance_data_times",
    "prepare_binned_classification_data_times",
    "create_performance_table",
    "create_performance_table_times",
    "render_performance_table",
    "create_summary_report",
}


def test_all_lists_every_top_level_public_api():
    assert set(rtichoke.__all__) == EXPECTED_PUBLIC_EXPORTS

    for name in rtichoke.__all__:
        assert hasattr(rtichoke, name)

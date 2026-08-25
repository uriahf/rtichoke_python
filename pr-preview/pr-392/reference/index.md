# API Reference


## Performance Data


Prepare classification and time-to-event data for visualization.


[prepare_performance_data()](prepare_performance_data.md#rtichoke.prepare_performance_data)  
Prepare performance data for binary classification models.

[prepare_binned_classification_data()](prepare_binned_classification_data.md#rtichoke.prepare_binned_classification_data)  
Prepare probability-binned classification data for binary outcomes.

[prepare_performance_data_times()](prepare_performance_data_times.md#rtichoke.prepare_performance_data_times)  
Prepare performance data for models with time-to-event outcomes.

[prepare_binned_classification_data_times()](prepare_binned_classification_data_times.md#rtichoke.prepare_binned_classification_data_times)  
Prepare binned, time-dependent classification data.


## Performance Tables


Summarize model performance across thresholds and time horizons.


[create_performance_table()](create_performance_table.md#rtichoke.create_performance_table)  
Create an R-style rtichoke performance table.

[create_performance_table_times()](create_performance_table_times.md#rtichoke.create_performance_table_times)  
Create a time-dependent rtichoke performance table.

[render_performance_table()](render_performance_table.md#rtichoke.render_performance_table)  
Render prepared performance data with a selected table backend.


## Discrimination


ROC, precision-recall, gains, and lift visualizations.


[create_roc_curve()](create_roc_curve.md#rtichoke.create_roc_curve)  
Creates a Receiver Operating Characteristic (ROC) curve.

[create_roc_curve_times()](create_roc_curve_times.md#rtichoke.create_roc_curve_times)  
Creates a time-dependent Receiver Operating Characteristic (ROC) curve.

[plot_roc_curve()](plot_roc_curve.md#rtichoke.plot_roc_curve)  
Plots an ROC curve from pre-computed performance data.

[create_precision_recall_curve()](create_precision_recall_curve.md#rtichoke.create_precision_recall_curve)  
Creates a Precision-Recall curve.

[create_precision_recall_curve_times()](create_precision_recall_curve_times.md#rtichoke.create_precision_recall_curve_times)  
Creates a time-dependent Precision-Recall curve.

[plot_precision_recall_curve()](plot_precision_recall_curve.md#rtichoke.plot_precision_recall_curve)  
Plots a Precision-Recall curve from pre-computed performance data.

[create_gains_curve()](create_gains_curve.md#rtichoke.create_gains_curve)  
Creates a Gains curve.

[create_gains_curve_times()](create_gains_curve_times.md#rtichoke.create_gains_curve_times)  
Creates a time-dependent Gains curve.

[plot_gains_curve()](plot_gains_curve.md#rtichoke.plot_gains_curve)  
Plots a Gains curve from pre-computed performance data.

[create_lift_curve()](create_lift_curve.md#rtichoke.create_lift_curve)  
Creates a Lift curve.

[create_lift_curve_times()](create_lift_curve_times.md#rtichoke.create_lift_curve_times)  
Creates a time-dependent Lift curve.

[plot_lift_curve()](plot_lift_curve.md#rtichoke.plot_lift_curve)  
Plots a Lift curve from pre-computed performance data.


## Calibration


Calibration visualizations for classification and time-to-event models. See Curve API Compatibility for time-dependent heuristic and horizon differences.


[create_calibration_curve()](create_calibration_curve.md#rtichoke.create_calibration_curve)  
Create an interactive calibration plot with a square main panel.

[create_calibration_curve_times()](create_calibration_curve_times.md#rtichoke.create_calibration_curve_times)  
Create an interactive time-dependent calibration plot with a square main panel.


## Summary Reports


Create historical R-backed reports or explicitly opt into canonical browser ReportSpec rendering.


[create_summary_report()](create_summary_report.md#rtichoke.create_summary_report)  
Create an rtichoke model-performance summary report.


## Utility


Decision-curve analysis for classification and time-to-event models.


[create_decision_curve()](create_decision_curve.md#rtichoke.create_decision_curve)  
Creates a Decision Curve.

[create_decision_curve_times()](create_decision_curve_times.md#rtichoke.create_decision_curve_times)  
Creates a time-dependent Decision Curve using the existing Plotly path.

[plot_decision_curve()](plot_decision_curve.md#rtichoke.plot_decision_curve)  
Plots a Decision Curve from pre-computed performance data.

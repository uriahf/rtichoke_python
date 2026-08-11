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
Creates Calibration Curve

[create_calibration_curve_times()](create_calibration_curve_times.md#rtichoke.create_calibration_curve_times)  
Creates a time-dependent Calibration Curve with a slider for different time horizons.


## Utility


Decision-curve analysis for classification and time-to-event models.


[create_decision_curve()](create_decision_curve.md#rtichoke.create_decision_curve)  
Creates a Decision Curve.

[create_decision_curve_times()](create_decision_curve_times.md#rtichoke.create_decision_curve_times)  
Creates a time-dependent Decision Curve.

[plot_decision_curve()](plot_decision_curve.md#rtichoke.plot_decision_curve)  
Plots a Decision Curve from pre-computed performance data.

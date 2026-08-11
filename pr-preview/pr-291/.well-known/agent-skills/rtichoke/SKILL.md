---
name: rtichoke
description: >
  interactive visualizations for performance of predictive models. Use when writing Python code that uses the rtichoke package.
license: MIT
compatibility: Requires Python >=3.9.
---

# rtichoke

interactive visualizations for performance of predictive models

## Installation

```bash
pip install rtichoke
```

## API overview

### Performance Data

Prepare classification and time-to-event data for visualization.

- `prepare_performance_data`: Prepare performance data for binary classification models
- `prepare_binned_classification_data`: Prepare probability-binned classification data for binary outcomes
- `prepare_performance_data_times`: Prepare performance data for models with time-to-event outcomes
- `prepare_binned_classification_data_times`: Prepare binned, time-dependent classification data

### Discrimination

ROC, precision-recall, gains, and lift visualizations.

- `create_roc_curve`: Creates a Receiver Operating Characteristic (ROC) curve
- `create_roc_curve_times`: Creates a time-dependent Receiver Operating Characteristic (ROC) curve
- `plot_roc_curve`: Plots an ROC curve from pre-computed performance data
- `create_precision_recall_curve`: Creates a Precision-Recall curve
- `create_precision_recall_curve_times`: Creates a time-dependent Precision-Recall curve
- `plot_precision_recall_curve`: Plots a Precision-Recall curve from pre-computed performance data
- `create_gains_curve`: Creates a Gains curve
- `create_gains_curve_times`: Creates a time-dependent Gains curve
- `plot_gains_curve`: Plots a Gains curve from pre-computed performance data
- `create_lift_curve`: Creates a Lift curve
- `create_lift_curve_times`: Creates a time-dependent Lift curve
- `plot_lift_curve`: Plots a Lift curve from pre-computed performance data

### Calibration

Calibration visualizations for classification and time-to-event models.

- `create_calibration_curve`: Create a calibration curve
- `create_calibration_curve_times`: Create time-dependent calibration curves

### Utility

Decision-curve analysis for classification and time-to-event models.

- `create_decision_curve`: Creates a Decision Curve
- `create_decision_curve_times`: Creates a time-dependent Decision Curve
- `plot_decision_curve`: Plots a Decision Curve from pre-computed performance data

## Resources

- [Full documentation](https://uriahf.github.io/rtichoke_python/)
- [llms.txt](llms.txt) — Indexed API reference for LLMs
- [llms-full.txt](llms-full.txt) — Comprehensive documentation for LLMs

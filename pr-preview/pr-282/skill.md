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

Functions for preparing model-performance data.

- `prepare_performance_data`: Prepare performance data for binary classification models
- `prepare_performance_data_times`: Prepare performance data for models with time-to-event outcomes

### Discrimination

Functions for evaluating discrimination.

- `create_roc_curve`: Creates a Receiver Operating Characteristic (ROC) curve
- `plot_roc_curve`: Plots an ROC curve from pre-computed performance data
- `create_precision_recall_curve`: Creates a Precision-Recall curve
- `plot_precision_recall_curve`: Plots a Precision-Recall curve from pre-computed performance data
- `create_gains_curve`: Creates a Gains curve
- `plot_gains_curve`: Plots a Gains curve from pre-computed performance data
- `create_lift_curve`: Creates a Lift curve
- `plot_lift_curve`: Plots a Lift curve from pre-computed performance data

### Calibration and Utility

Functions for calibration and decision-curve analysis.

- `create_calibration_curve`: Creates Calibration Curve
- `create_decision_curve`: Creates a Decision Curve
- `plot_decision_curve`: Plots a Decision Curve from pre-computed performance data

### Reports

Summary reporting helpers.

- `create_summary_report`: Create rtichoke Summary Report

## Resources

- [Full documentation](https://uriahf.github.io/rtichoke_python/)
- [llms.txt](llms.txt) — Indexed API reference for LLMs
- [llms-full.txt](llms-full.txt) — Comprehensive documentation for LLMs

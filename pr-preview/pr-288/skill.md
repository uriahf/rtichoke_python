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

- `prepare_performance_data`
- `prepare_binned_classification_data`
- `prepare_performance_data_times`
- `prepare_binned_classification_data_times`

### Discrimination

ROC, precision-recall, gains, and lift visualizations.

- `create_roc_curve`
- `create_roc_curve_times`
- `plot_roc_curve`
- `create_precision_recall_curve`
- `create_precision_recall_curve_times`
- `plot_precision_recall_curve`
- `create_gains_curve`
- `create_gains_curve_times`
- `plot_gains_curve`
- `create_lift_curve`
- `create_lift_curve_times`
- `plot_lift_curve`

### Calibration

Calibration visualizations for classification and time-to-event models.

- `create_calibration_curve`
- `create_calibration_curve_times`

### Utility

Decision-curve analysis for classification and time-to-event models.

- `create_decision_curve`
- `create_decision_curve_times`
- `plot_decision_curve`

## Resources

- [Full documentation](https://uriahf.github.io/rtichoke_python/)
- [llms.txt](llms.txt) — Indexed API reference for LLMs
- [llms-full.txt](llms-full.txt) — Comprehensive documentation for LLMs

# Summary reports

[create_summary_report()](../reference/create_summary_report.md#rtichoke.create_summary_report) keeps the historical R-backed report path as its default. The canonical browser report is available only when explicitly requested with `renderer="browser"`.

``` python
import numpy as np
from rtichoke import create_summary_report

probs = {
    "Model A": np.array(
        [0.03, 0.08, 0.12, 0.18, 0.25, 0.32, 0.40, 0.50, 0.62, 0.75, 0.88, 0.96]
    )
}
reals = np.array([0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1])

create_summary_report(
    probs,
    reals,
    renderer="browser",
    output_file="summary_report.html",
)
```

The browser path uses the same production calculations as the standalone Python components, converts those results with existing canonical component builders, assembles a canonical ReportSpec, and delegates report composition to the vendored immutable `rtichoke_viz` renderer.

The Summary Report browser backend writes a single self-contained, offline-ready HTML artifact with all required shared renderer JavaScript and CSS embedded directly into the file. The browser backend returns the generated HTML `pathlib.Path`; the default historical R backend retains its existing `None` return behavior.


# Static browser Summary Report

The static browser Summary Report contains six top-level sections in the following exact order:

1.  **Prevalence**: Displays population prevalence summary metrics.
2.  **Prediction Distribution**: Displays decomposed predicted probability histograms with interactive operating-point controls, offered in two views/groups:
    - **By Probability Threshold**
    - **By Predicted Positives Condition Rate (PPCR)**
3.  **Calibration**: Displays model calibration curves in two views:
    - **Smooth**
    - **Discrete**
4.  **Discrimination**: Displays summary metrics (AUROC) and curve visualizations across both operating-point dimensions (**By Probability Threshold** and **By Predicted Positives Condition Rate (PPCR)**) in exact component order:
    - **ROC**
    - **Lift**
    - **Precision-Recall**
    - **Gains**
5.  **Utility**: Displays Decision Curve and Interventions Avoided curves.
6.  **Performance Table**: Displays full performance metrics across operating points, grouped **By Probability Threshold** and **By Predicted Positives Condition Rate (PPCR)**.


# Time-dependent Summary Report

For survival and time-to-event outcomes, `create_summary_report_times()` generates a canonical time-dependent browser summary report at fixed time horizons:

``` python
import numpy as np
from rtichoke import create_summary_report_times

probs = {
    "Model A": np.array(
        [0.03, 0.08, 0.12, 0.18, 0.25, 0.32, 0.40, 0.50, 0.62, 0.75, 0.88, 0.96]
    )
}
reals = np.array([0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1])
times = np.array([10, 12, 15, 8, 20, 25, 30, 18, 22, 14, 19, 28])

create_summary_report_times(
    probs,
    reals,
    times,
    fixed_time_horizons=[15.0],
    output_file="summary_report_times.html",
)
```

The time-dependent browser report contains five top-level sections in exact order:

1.  **Event Probability**: Displays population event probabilities at specified fixed time horizons (derived using time-dependent estimators accounting for censoring and competing events).
2.  **Calibration**: Displays smooth and discrete time-dependent calibration curves.
3.  **Discrimination**: Displays time-dependent curve visualizations grouped **By Probability Threshold** and **By Predicted Positives Condition Rate (PPCR)** in exact component order:
    - **ROC**
    - **Lift**
    - **Precision-Recall**
    - **Gains**
4.  **Utility**: Displays time-dependent Decision Curve and Interventions Avoided curves.
5.  **Performance Table**: Displays time-dependent performance tables grouped **By Probability Threshold** and **By Predicted Positives Condition Rate (PPCR)**.


## Architectural differences from static reports

The time-dependent summary report intentionally differs from the binary static report in three key ways:

- **No Prevalence section**: Replaced by the time-horizon **Event Probability** section to account for censoring and competing events over time.
- **No Prediction Distribution section**: Prediction distribution histograms are omitted in time-dependent reports.
- **No AUROC summary metric**: AUROC summary metric cards are omitted in time-dependent reports.

The browser backend does not replace Quarto or the historical R backend, and `renderer="browser"` is an opt-in path. Existing Plotly, Matplotlib, table, and standalone browser-chart APIs remain unchanged.

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

The browser path uses the same production calculations as the standalone Python components, converts those results with the existing canonical component builders, assembles a canonical ReportSpec, and delegates report composition to the vendored immutable `rtichoke_viz v0.5.0` `renderReport()` implementation.

The first public browser report contains, in deterministic order:

1.  PerformanceTable;
2.  ROC-v2;
3.  calibration-v2.

The generated HTML is accompanied by `rtichoke-viz.js` and `rtichoke-viz.css` in the same directory, so those three files should be kept together when moving the report. The browser backend returns the generated HTML `pathlib.Path`; the default historical R backend retains its existing `None` return behavior.

The browser backend does not replace Quarto or the historical R backend, and it is not the default. Existing Plotly, Matplotlib, table, standalone browser-chart, and time-dependent APIs are unchanged by this opt-in report path.

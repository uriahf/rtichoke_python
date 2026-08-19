# D3 summary report proof of concept

This branch replaces the old R-backend `create_summary_report()` stub with a native Python binary report. It prepares the Polars performance table once and serializes only the columns needed by five D3 panels. The PR preview workflow publishes the generated example as `summary-report-demo.html`.

The first preview deliberately uses the D3 CDN so the architecture and interaction can be reviewed before vendoring D3 into the package. A production version should bundle D3 to make the output genuinely self-contained.

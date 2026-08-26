## create_decision_curve_times()


Creates a time-dependent Decision Curve.


Usage

``` python
create_decision_curve_times(
    probs,
    reals,
    times,
    fixed_time_horizons,
    decision_type="conventional",
    heuristics_sets=[{"censoring_heuristic": "adjusted", "competing_heuristic": "adjusted_as_negative"}],
    min_p_threshold=0,
    max_p_threshold=1,
    by=0.01,
    stratified_by=["probability_threshold"],
    size=600,
    color_values=["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#07004D", "#E6AB02", "#FE5F55", "#54494B", "#006E90", "#BC96E6", "#52050A", "#1F271B", "#BE7C4D", "#63768D", "#08A045", "#320A28", "#82FF9E", "#2176FF", "#D1603D", "#585123"],
    renderer="plotly"
)
```


`renderer="plotly"` preserves the historical default. For time-dependent conventional Decision Curves, `"browser"` and `"rtichoke_viz"` return a canonical `RtichokeBrowserChart` built from already-computed production values.

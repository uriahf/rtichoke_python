## plot_decision_curve()


Plots a Decision Curve from pre-computed performance data.


Usage

``` python
plot_decision_curve(
    performance_data,
    decision_type="conventional",
    min_p_threshold=0,
    max_p_threshold=1,
    stratified_by=["probability_threshold"],
    size=600,
    renderer="plotly"
)
```


For browser rendering, pre-computed `reference_group` values are treated as distinct populations because separate model identity is not encoded in this input shape.

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
    size=600
)
```


This function is useful for plotting a Decision Curve directly from a DataFrame that already contains the necessary performance metrics.


## Parameters


`performance_data: pl.DataFrame`  
A Polars DataFrame with performance metrics, including net benefit and probability thresholds.

`decision_type: str = ``"conventional"`  
Type of decision curve to plot. Defaults to `"conventional"`.

`min_p_threshold: float = ``0`  
The minimum probability threshold to plot. Defaults to 0.

`max_p_threshold: float = ``1`  
The maximum probability threshold to plot. Defaults to 1.

`stratified_by: Sequence[str] = [`<span class="st">`"probability_threshold"]`\
</span>  
The columns in `performance_data` used for stratification. Defaults to `["probability_threshold"]`.

`size: int = ``600`  
The width and height of the plot in pixels. Defaults to 600.


## Returns


`Figure`  
A Plotly `Figure` object representing the Decision Curve.

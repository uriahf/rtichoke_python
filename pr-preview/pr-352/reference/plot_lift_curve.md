## plot_lift_curve()


Plots a Lift curve from pre-computed performance data.


Usage

``` python
plot_lift_curve(
    performance_data, stratified_by=["probability_threshold"], size=600
)
```


This function is useful for plotting a Lift curve directly from a DataFrame that already contains the necessary performance metrics.


## Parameters


`performance_data: pl.DataFrame`  
A Polars DataFrame with performance metrics. It must include columns for the lift values and the percentage of the population targeted, along with any stratification variables.

`stratified_by: Sequence[str] = [``"probability_threshold"]`  
The columns in `performance_data` used for stratification. Defaults to `["probability_threshold"]`.

`size: int = ``600`  
The width and height of the plot in pixels. Defaults to 600.


## Returns


`Figure`  
A Plotly `Figure` object representing the Lift curve.

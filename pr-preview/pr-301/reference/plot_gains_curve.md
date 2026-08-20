## plot_gains_curve()


Plots a Gains curve from pre-computed performance data.


Usage

``` python
plot_gains_curve(
    performance_data, stratified_by=["probability_threshold"], size=600
)
```


This function is useful for plotting a Gains curve directly from a DataFrame that already contains the necessary performance metrics.


## Parameters


`performance_data: pl.DataFrame`  
A Polars DataFrame with performance metrics. It must include columns for the percentage of the population targeted and the corresponding gain, along with any stratification variables.

`stratified_by: Sequence[str] = [``"probability_threshold"]`  
The columns in `performance_data` used for stratification. Defaults to `["probability_threshold"]`.

`size: int = ``600`  
The width and height of the plot in pixels. Defaults to 600.


## Returns


`Figure`  
A Plotly `Figure` object representing the Gains curve.

## plot_precision_recall_curve()


Plots a Precision-Recall curve from pre-computed performance data.


Usage

``` python
plot_precision_recall_curve(
    performance_data,
    stratified_by=["probability_threshold"],
    size=600,
    renderer="plotly"
)
```


This function is useful when you have already computed the performance metrics and want to generate a Precision-Recall plot directly. Pre-computed data does not encode separate model identity, so canonical browser rendering treats each `reference_group` as a population with unknown model identity.


## Parameters


`performance_data: pl.DataFrame`  
A Polars DataFrame with the necessary performance metrics, including precision (ppv) and recall (sensitivity), along with the production prevalence quantities `real_positives` and `n`.

`stratified_by: Sequence[str] = [`<span class="st">`"probability_threshold"]`\
</span>  
The columns in `performance_data` used for stratification. Defaults to `["probability_threshold"]`.

`size: int = ``600`  
The width and height of the plot in pixels. Defaults to 600.

`renderer: (plotly, browser, rtichoke_viz) = ``"plotly"`  
Rendering backend. `"plotly"` remains the default.


## Returns


`Figure or RtichokeBrowserChart`  
A Plotly `Figure` or canonical offline browser chart.

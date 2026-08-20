## plot_precision_recall_curve()


Plots a Precision-Recall curve from pre-computed performance data.


Usage

``` python
plot_precision_recall_curve(
    performance_data, stratified_by=["probability_threshold"], size=600
)
```


This function is useful when you have already computed the performance metrics and want to generate a Precision-Recall plot directly.


## Parameters


`performance_data: pl.DataFrame`  
A Polars DataFrame with the necessary performance metrics, including precision (ppv) and recall (tpr), along with any stratification variables.

`stratified_by: Sequence[str] = [`<span class="st">`"probability_threshold"]`\
</span>  
The columns in `performance_data` used for stratification. Defaults to `["probability_threshold"]`.

`size: int = ``600`  
The width and height of the plot in pixels. Defaults to 600.


## Returns


`Figure`  
A Plotly `Figure` object representing the Precision-Recall curve.

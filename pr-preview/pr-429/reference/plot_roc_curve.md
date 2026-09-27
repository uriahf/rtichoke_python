## plot_roc_curve()


Plots an ROC curve from pre-computed performance data.


Usage

``` python
plot_roc_curve(
    performance_data, stratified_by=["probability_threshold"], size=600
)
```


This function is useful when you have already computed the performance metrics (TPR, FPR, etc.) and want to generate an ROC plot directly from that data.


## Parameters


`performance_data: pl.DataFrame`  
A Polars DataFrame containing the necessary performance metrics. It must include columns for the true positive rate (tpr) and false positive rate (fpr), along with any stratification variables.

`stratified_by: Sequence[str] = [`<span class="st">`"probability_threshold"]`\
</span>  
The columns in `performance_data` used for stratification. Defaults to `["probability_threshold"]`.

`size: int = ``600`  
The width and height of the plot in pixels. Defaults to 600.


## Returns


`Figure`  
A Plotly `Figure` object representing the ROC curve.

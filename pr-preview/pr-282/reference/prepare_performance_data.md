## prepare_performance_data()


Prepare performance data for binary classification models.


Usage

``` python
prepare_performance_data(
    probs, reals, stratified_by=("probability_threshold",), by=0.01
)
```


This function computes a comprehensive set of performance metrics for one or more binary classification models across a range of probability thresholds. It builds upon the binned data from [prepare_binned_classification_data](prepare_binned_classification_data.md#rtichoke.prepare_binned_classification_data) by cumulatively summing the counts and calculating metrics like sensitivity (TPR), specificity, precision (PPV), and net benefit.

This resulting dataframe is the primary input for plotting functions like [plot_roc_curve](plot_roc_curve.md#rtichoke.plot_roc_curve), [plot_precision_recall_curve](plot_precision_recall_curve.md#rtichoke.plot_precision_recall_curve), etc.


## Parameters


`probs: Dict[str, np.ndarray]`  
A dictionary mapping model or dataset names (str) to their predicted probabilities (1-D numpy arrays).

`reals: Union[np.ndarray, Dict[str, np.ndarray]]`  
The true event labels. This can be a single numpy array that is aligned with all pooled probabilities or a dictionary mapping each dataset name to its corresponding array of true labels. Labels must be binary (0 or 1).

`stratified_by: Sequence[str] = (`<span class="st">`"probability_threshold",)`\
</span>  
A sequence of strings specifying the variables by which to stratify the data. The default is `("probability_threshold",)`.

`by: float = ``0.01`  
The step size for probability thresholds, determining the number of points at which performance is evaluated. Defaults to `0.01`.


## Returns


`pl.DataFrame`  
A Polars DataFrame where each row corresponds to a probability cutoff for a given model/dataset. Columns include the cutoff value and a rich set of performance metrics (e.g., `tpr`, `fpr`, `ppv`, `net_benefit`).


## Examples

``` python
>>> import numpy as np
>>> probs_dict_test = {
...     "small_data_set": np.array(
...         [0.9, 0.85, 0.95, 0.88, 0.6, 0.7, 0.51, 0.2, 0.1, 0.33]
...     )
... }
>>> reals_dict_test = [1, 1, 1, 1, 0, 0, 1, 0, 0, 1]
>>> performance_df = prepare_performance_data(
...     probs=probs_dict_test,
...     reals=reals_dict_test,
...     by=0.1
... )
```

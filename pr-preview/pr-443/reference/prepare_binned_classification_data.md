## prepare_binned_classification_data()


Prepare probability-binned classification data for binary outcomes.


Usage

``` python
prepare_binned_classification_data(
    probs, reals, stratified_by=("probability_threshold",), by=0.01
)
```


This function serves as the foundation for many of the performance analysis visualizations. It takes predicted probabilities and true binary outcomes, bins them by probability thresholds, and calculates the number of true positives, false positives, true negatives, and false negatives within each bin. This detailed, binned data can then be used to generate calibration plots or be aggregated to compute various performance metrics.


## Parameters


`probs: Dict[str, np.ndarray]`  
A dictionary mapping model or dataset names (str) to their predicted probabilities (1-D numpy arrays).

`reals: Union[np.ndarray, Dict[str, np.ndarray]]`  
The true event labels. This can be a single numpy array that is aligned with all pooled probabilities or a dictionary mapping each dataset name to its corresponding array of true labels. Labels must be binary (0 or 1).

`stratified_by: Sequence[str] = (`<span class="st">`"probability_threshold",)`\
</span>  
A sequence of strings specifying the variables by which to stratify the data. The default is `("probability_threshold",)`, which bins the data based on predicted probabilities.

`by: float = ``0.01`  
The step size to use when creating bins for the probability thresholds. This determines the granularity of the analysis. Defaults to `0.01`.


## Returns


`pl.DataFrame`  
A Polars DataFrame containing the binned classification data. Each row represents a unique combination of model/dataset, probability bin, and any other stratification variables. It forms the basis for subsequent performance calculations.

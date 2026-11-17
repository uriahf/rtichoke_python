## prepare_binned_classification_data_times()


Prepare binned, time-dependent classification data.


Usage

``` python
prepare_binned_classification_data_times(
    probs,
    reals,
    times,
    fixed_time_horizons,
    heuristics_sets=[{"censoring_heuristic": "adjusted", "competing_heuristic": "adjusted_as_negative"}],
    stratified_by=("probability_threshold",),
    by=0.01,
    risk_set_scope=["pooled_by_cutoff", "within_stratum"]
)
```


This function constructs the foundational binned data needed for time-to-event performance analysis. It bins predictions by probability thresholds, applies censoring and competing event heuristics, and stratifies the data across specified time horizons. The output is a detailed breakdown of outcomes within each bin, which can be used for calibration or passed to [prepare_performance_data_times](prepare_performance_data_times.md#rtichoke.prepare_performance_data_times) for full performance metric calculation.


## Parameters


`probs: Dict[str, np.ndarray]`  
A dictionary mapping model or dataset names (str) to their predicted probabilities.

`reals: Union[np.ndarray, Dict[str, np.ndarray]]`  
The true event statuses (e.g., 0=censored, 1=event, 2=competing event).

`times: Union[np.ndarray, Dict[str, np.ndarray]]`  
The event or censoring times.

`fixed_time_horizons: list[float]`  
A list of numeric time points for performance evaluation. Integer inputs are accepted and normalized to floats.

`heuristics_sets: list[Dict] = [{``"censoring_heuristic": `<span class="st">`"adjusted"``, ``"competing_heuristic"``: ``"adjusted_as_negative"``}]`\
</span>  
Specifies how to handle censored data and competing events.

`stratified_by: Sequence[str] = (`<span class="st">`"probability_threshold",)`\
</span>  
Variables for stratification. Defaults to `("probability_threshold",)`.

`by: float = ``0.01`  
The step size for probability thresholds. Defaults to `0.01`.

`risk_set_scope: Sequence[str] = [``"pooled_by_cutoff", `<span class="st">`"within_stratum"``]`\
</span>  
Defines the scope for risk set calculations. Defaults to `["pooled_by_cutoff", "within_stratum"]`.


## Returns


`pl.DataFrame`  
A Polars DataFrame with binned, time-dependent data. Each row represents a unique combination of dataset, bin, time horizon, heuristic, and other strata.

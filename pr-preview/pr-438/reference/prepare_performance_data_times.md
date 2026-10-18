## prepare_performance_data_times()


Prepare performance data for models with time-to-event outcomes.


Usage

``` python
prepare_performance_data_times(
    probs,
    reals,
    times,
    fixed_time_horizons,
    heuristics_sets=[{"censoring_heuristic": "adjusted", "competing_heuristic": "adjusted_as_negative"}],
    stratified_by=("probability_threshold",),
    by=0.01
)
```


This function calculates a comprehensive set of performance metrics for models predicting time-to-event outcomes. It handles censored data and competing events by applying specified heuristics at different time horizons. The function first bins the data using [prepare_binned_classification_data_times](prepare_binned_classification_data_times.md#rtichoke.prepare_binned_classification_data_times) and then computes cumulative, Aalen-Johansen-based performance metrics.

The resulting dataframe is the primary input for time-dependent plotting functions.


## Parameters


`probs: Dict[str, np.ndarray]`  
A dictionary mapping model or dataset names (str) to their predicted probabilities of an event occurring by a given time.

`reals: Union[np.ndarray, Dict[str, np.ndarray]]`  
The true event statuses. Can be a single array or a dictionary. Labels should be integers indicating the outcome (e.g., 0=censored, 1=event of interest, 2=competing event).

`times: Union[np.ndarray, Dict[str, np.ndarray]]`  
The event or censoring times corresponding to the `reals`. Can be a single array or a dictionary.

`fixed_time_horizons: list[float]`  
A list of numeric time points at which to evaluate the model's performance. Integer inputs are accepted and normalized to floats.

`heuristics_sets: list[Dict] = [{``"censoring_heuristic": `<span class="st">`"adjusted"``, ``"competing_heuristic"``: ``"adjusted_as_negative"``}]`\
</span>  
A list of dictionaries, each specifying how to handle censored data and competing events. The default is `[{"censoring_heuristic": "adjusted", "competing_heuristic": "adjusted_as_negative"}]`.

`stratified_by: Sequence[str] = (`<span class="st">`"probability_threshold",)`\
</span>  
Variables by which to stratify the analysis. Defaults to `("probability_threshold",)`.

`by: float = ``0.01`  
The step size for probability thresholds. Defaults to `0.01`.


## Returns


`pl.DataFrame`  
A Polars DataFrame with performance metrics computed across probability thresholds and time horizons. It includes columns for cutoffs, time points, heuristics, and performance measures.

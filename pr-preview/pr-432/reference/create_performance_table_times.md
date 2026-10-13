## create_performance_table_times()


Create a time-dependent rtichoke performance table.


Usage

``` python
create_performance_table_times(
    probs,
    reals,
    times,
    fixed_time_horizons,
    heuristics_sets=_DEFAULT_HEURISTICS,
    by=0.01,
    stratified_by=("probability_threshold",),
    color_values=DEFAULT_COLORS,
    renderer="great_tables"
)
```


Numerical results come from [prepare_performance_data_times()](prepare_performance_data_times.md#rtichoke.prepare_performance_data_times). The table keeps time horizon and censoring/competing-event heuristics visible so that multiple requested evaluation scenarios are not collapsed in presentation. Observed times are normalized to floating point at this public wrapper boundary; fixed-horizon normalization is handled by the shared time-dependent performance pipeline.

## create_calibration_curve_times()


Create time-dependent calibration curves.


Usage

``` python
create_calibration_curve_times(
    probs,
    reals,
    times,
    fixed_time_horizons,
    heuristics_sets,
    calibration_type="discrete",
    size=600,
    color_values=_DEFAULT_COLORS
)
```


## Parameters


`probs: dict[str, numpy.ndarray]`  
Predicted probabilities. Matching dictionary keys identify populations.

`reals: numpy.ndarray or dict[str, numpy.ndarray]`  
Observed event indicators.

`times: numpy.ndarray or dict[str, numpy.ndarray]`  
Observed event or censoring times.

`fixed_time_horizons: list[float]`  
Time horizons to display.

`heuristics_sets: list[dict[str, str]]`  
Censoring and competing-risk heuristic combinations.

`calibration_type: str = ``"discrete"`  
Calibration rendering type.

`size: int = ``600`  
Figure width and height in pixels.

`color_values: list[str] = _DEFAULT_COLORS`\  
Colors used for population or model traces.


## Returns


`plotly.graph_objs.Figure`  
Interactive time-dependent calibration figure.

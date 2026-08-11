## create_calibration_curve()


Create a calibration curve.


Usage

``` python
create_calibration_curve(
    probs,
    reals,
    calibration_type="discrete",
    size=600,
    color_values=_DEFAULT_COLORS
)
```


## Parameters


`probs: dict[str, numpy.ndarray]`  
Predicted probabilities. When `reals` is a dictionary with matching keys, each key is treated as an independent population and may have its own sample size.

`reals: numpy.ndarray or dict[str, numpy.ndarray]`  
Observed binary outcomes. Matching dictionary keys are paired population-by-population.

`calibration_type: str = ``"discrete"`  
Calibration rendering type, either `"discrete"` or `"smooth"`.

`size: int = ``600`  
Figure width and height in pixels.

`color_values: list[str] = _DEFAULT_COLORS`\  
Colors used for population or model traces.


## Returns


`plotly.graph_objs.Figure`  
Interactive calibration figure.

## create_reals_summary_report_times()


Create a minimal time-dependent outcome distribution browser summary report.


Usage

``` python
create_reals_summary_report_times(
    reals,
    times,
    fixed_time_horizons,
    *,
    renderer="browser",
    output_file="summary_report_times.html",
    heuristics=False
)
```


The report contains only Outcome Distribution, without predictions or model-performance metrics. By default it displays raw observed statuses. Enable `heuristics` for eight interactive censoring/competing-event combinations using the existing Kaplan-Meier/Aalen-Johansen preparation.


## Parameters


`reals: Union[np.ndarray, list[int], Dict[str, Any]]`  
Outcome labels: 0 = censored, 1 = primary event, 2 = competing event. Dictionary keys name distinct populations and must match `times` in both keys and order. Each population must be nonempty.

`times: Union[np.ndarray, list[float], Dict[str, Any]]`  
Follow-up times. Target and competing events require positive times. Censoring may occur at time zero.

`fixed_time_horizons: list[float]`  
Nonempty finite nonnegative horizons; sorted, deduplicated, with zero included. Events at the horizon are included; censoring at the horizon remains event-free when censoring is excluded.

`renderer: (``"browser", `<span class="st">`"rtichoke_viz"``)`</span>` = ``"browser"`  
Renderer backend. Defaults to `"browser"`.

`output_file: str or pathlib.Path = ``"summary_report_times.html"`  
HTML destination file path. Defaults to `"summary_report_times.html"`.

`heuristics: bool = ``False`  
Enable censoring and competing-event controls. Defaults to `False`. The initial selection adjusts censoring and treats competing events as negative. Exclusions define a retained population at each horizon; exclusion percentages use original population size. Empty retained populations have unavailable estimates. Beyond follow-up support, displayed estimates retain the estimator's last plateau. Adjusted estimates require the relevant independent-censoring assumptions.


## Returns


`pathlib.Path`  
The generated HTML file path.


## Examples

``` python
>>> create_reals_summary_report_times(
...     reals=[1, 0, 2], times=[2, 5, 4], fixed_time_horizons=[3, 5],
...     output_file="outcomes.html", heuristics=True,
... )
```

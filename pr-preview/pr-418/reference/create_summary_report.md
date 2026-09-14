## create_summary_report()


Create an rtichoke model-performance summary report.


Usage

``` python
create_summary_report(
    probs,
    reals,
    url_api="http://localhost:4242/",
    *,
    renderer="r",
    output_file="summary_report.html"
)
```


The default `renderer="r"` preserves the historical public behavior and delegates to the R rtichoke backend at `url_api`. `renderer="browser"` is an explicit opt-in path that uses Python's existing production calculations, canonical standalone component builders, canonical ReportSpec assembly, and the vendored `rtichoke_viz` `renderReport()` composer.


## Parameters


`probs: Dict[str, np.ndarray]`  
A dictionary mapping model or population names to predicted probabilities.

`reals: Union[np.ndarray, Dict[str, np.ndarray]]`  
The true binary outcome labels.

`url_api: str = ``"http://localhost:4242/"`  
The API endpoint URL of the historical R rtichoke backend. Used only by `renderer="r"`. Defaults to `"http://localhost:4242/"`.

`renderer: (``"r", `<span class="st">`"browser"``)`</span>` = ``"r"`  
Summary-report backend. Defaults to `"r"` for backward compatibility.

`output_file: str or pathlib.Path = ``"summary_report.html"`  
HTML destination for `renderer="browser"`. Defaults to `"summary_report.html"`.


## Returns


`pathlib.Path or None`  
The generated HTML path for `renderer="browser"`; `None` for the historical R backend.

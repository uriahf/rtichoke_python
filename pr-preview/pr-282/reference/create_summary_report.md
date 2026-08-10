## create_summary_report()


Request an rtichoke summary report from the rtichoke R API.


Usage

``` python
create_summary_report(
    probs,
    reals,
    url_api="http://localhost:4242/",
)
```


The current implementation sends model predictions and observed outcomes to the [create_summary_report](create_summary_report.md#rtichoke.create_summary_report) endpoint exposed by the rtichoke R service. It prints the keys returned by the service and does not currently return a Python report object or write a standalone HTML file.


## Parameters


`probs: dict`  
Predicted probabilities, typically keyed by model or population name.

`reals: dict`  
Observed binary outcomes, typically keyed by population name.

`url_api: str = ``"http://localhost:4242/"`  
Base URL of the running rtichoke R API service.


## Returns


`None`  
The function currently prints information from the API response.


## Notes

A running rtichoke R API service is required at `url_api`. This helper represents the current bridge to the R summary-report implementation; a self-contained Python-native HTML report is not yet implemented here.

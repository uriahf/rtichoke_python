# Getting Started

`rtichoke` is a Python library for interactive visualization of predictive-model performance. It supports discrimination, calibration, utility, and time-to-event evaluation workflows.

For some reproducible examples please visit [rtichoke blog](https://rtichoke-blog.netlify.app/)!


# Installation

If you use [uv](https://docs.astral.sh/uv/) to manage your Python project, add `rtichoke` with:

``` bash
uv add rtichoke
```

This adds `rtichoke` to your project dependencies and updates the uv lockfile.

If you are not using uv, install `rtichoke` from PyPI with pip:

``` bash
pip install rtichoke
```


# Import

``` python
import numpy as np
import rtichoke as rk
```


# Inputs

Most `rtichoke` plotting functions use two dictionaries:

- `probs`: model predictions, keyed by model or population name.
- `reals`: observed outcomes, keyed by population name.

> **Tip: Tip**
>
> Similar curve families can still differ in defaults and time-dependent handling. See [Curve API Compatibility](curve-api-compatibility.md), and if a call fails, search [Common Errors & Fixes](common-errors.md) by literal exception text.


# Single model

``` python
probs_single = {
    "Model A": np.array([0.1, 0.9, 0.4, 0.8, 0.3, 0.7, 0.2, 0.6])
}
reals_single = {
    "Population": np.array([0, 1, 0, 1, 0, 1, 0, 1])
}

fig = rk.create_roc_curve(
    probs=probs_single,
    reals=reals_single,
)

fig.show()
```


# Compare models

When several models are evaluated on the same population, provide one probability vector per model and one outcome vector for the shared population.

``` python
probs_comparison = {
    "Model A": np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7]),
    "Model B": np.array([0.2, 0.8, 0.3, 0.7, 0.4, 0.6]),
    "Random Guess": np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5]),
}
reals_comparison = {
    "Population": np.array([0, 1, 0, 1, 0, 1])
}

fig = rk.create_precision_recall_curve(
    probs=probs_comparison,
    reals=reals_comparison,
)

fig.show()
```


# Compare populations

To compare a model across populations, provide matching keys in `probs` and `reals`. Population sizes may differ; each probability vector only needs to match the outcome vector for the same key.

``` python
probs_populations = {
    "Train": np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7]),
    "Test": np.array([0.2, 0.8, 0.3, 0.7]),
}
reals_populations = {
    "Train": np.array([0, 1, 0, 1, 0, 1]),
    "Test": np.array([0, 1, 0, 0]),
}

fig = rk.create_calibration_curve(
    probs=probs_populations,
    reals=reals_populations,
)

fig.show()
```

Here, `Train` contains six observations and `Test` contains four. This matching-key contract is supported by calibration as well as the other curve families.

From here, use the API Reference for the full set of curve types, parameters, and time-to-event variants. The [Naming Conventions](naming-conventions.md) guide explains how the exported function families fit together, while [Curve API Compatibility](curve-api-compatibility.md) documents where those families still differ.


### Links

[View on PyPI![](data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxZW0iIGhlaWdodD0iMWVtIiBmaWxsPSJub25lIiBzdHJva2U9ImN1cnJlbnRDb2xvciIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiIHN0eWxlPSJ2ZXJ0aWNhbC1hbGlnbjogLTAuMDVlbTsgbWFyZ2luLWxlZnQ6IDBlbTsgbWFyZ2luLXRvcDogMC4xZW07IiB2aWV3Ym94PSIwIDAgMjQgMjQiPjxwYXRoIGQ9Ik03IDdoMTB2MTAiIC8+PHBhdGggZD0iTTcgMTcgMTcgNyIgLz48L3N2Zz4=)](https://pypi.org/project/rtichoke/)\


### AI / Agents

[Skills<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iZ2Qtc3BhcmtsZS1jdXJhdGVkIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIwLjg1ZW0iIGhlaWdodD0iMC44NWVtIiB2aWV3Ym94PSIwIDAgMjQgMjQiIGZpbGw9Im5vbmUiIHN0cm9rZT0iY3VycmVudENvbG9yIiBzdHJva2Utd2lkdGg9IjIiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIgc3Ryb2tlLWxpbmVqb2luPSJyb3VuZCIgc3R5bGU9InZlcnRpY2FsLWFsaWduOiAtMC4xZW07IG1hcmdpbi1sZWZ0OiAwLjI1ZW07Ij48cGF0aCBkPSJNOS45MzcgMTUuNUEyIDIgMCAwIDAgOC41IDE0LjA2M2wtNi4xMzUtMS41ODJhLjUuNSAwIDAgMSAwLS45NjJMOC41IDkuOTM2QTIgMiAwIDAgMCA5LjkzNyA4LjVsMS41ODItNi4xMzVhLjUuNSAwIDAgMSAuOTYzIDBMMTQuMDYzIDguNUEyIDIgMCAwIDAgMTUuNSA5LjkzN2w2LjEzNSAxLjU4MmEuNS41IDAgMCAxIDAgLjk2M0wxNS41IDE0LjA2M2EyIDIgMCAwIDAtMS40MzcgMS40MzdsLTEuNTgyIDYuMTM1YS41LjUgMCAwIDEtLjk2MyAweiIgLz48cGF0aCBkPSJNMjAgM3Y0IiAvPjxwYXRoIGQ9Ik0yMiA1aC00IiAvPjwvc3ZnPg==" class="gd-sparkle-curated" />](skills.md)\
[llms.txt](llms.txt)\
[llms-full.txt](llms-full.txt)\


### Developers


**Uriah Finkel**


### Community

[Contributing guide](./contributing.html)\


### Meta

**Requires:** Python `>=3.12`\
[Package Info](package-info.md)

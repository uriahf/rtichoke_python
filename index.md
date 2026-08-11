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

To compare a model across populations, provide matching keys in `probs` and `reals`.

``` python
probs_populations = {
    "Train": np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7]),
    "Test": np.array([0.2, 0.8, 0.3, 0.7, 0.4, 0.6]),
}
reals_populations = {
    "Train": np.array([0, 1, 0, 1, 0, 1]),
    "Test": np.array([0, 1, 0, 1, 0, 0]),
}

fig = rk.create_calibration_curve(
    probs=probs_populations,
    reals=reals_populations,
)

fig.show()
```

From here, use the API Reference for the full set of curve types, parameters, and time-to-event variants. The [Naming Conventions](user-guide/naming-conventions.html) guide explains how the exported function families fit together.


### Links

[View on PyPI![](data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxZW0iIGhlaWdodD0iMWVtIiBmaWxsPSJub25lIiBzdHJva2U9ImN1cnJlbnRDb2xvciIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiIHN0eWxlPSJ2ZXJ0aWNhbC1hbGlnbjogLTAuMDVlbTsgbWFyZ2luLWxlZnQ6IDBlbTsgbWFyZ2luLXRvcDogMC4xZW07IiB2aWV3Ym94PSIwIDAgMjQgMjQiPjxwYXRoIGQ9Ik03IDdoMTB2MTAiIC8+PHBhdGggZD0iTTcgMTcgMTcgNyIgLz48L3N2Zz4=)](https://pypi.org/project/rtichoke/)\


### AI / Agents

[Skills<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iZ2Qtc3BhcmtsZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB3aWR0aD0iMC44NWVtIiBoZWlnaHQ9IjAuODVlbSIgdmlld2JveD0iMCAwIDI0IDI0IiBmaWxsPSJub25lIiBzdHJva2U9ImN1cnJlbnRDb2xvciIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiIHN0eWxlPSJ2ZXJ0aWNhbC1hbGlnbjogLTAuMWVtOyBtYXJnaW4tbGVmdDogMC4yNWVtOyI+PHBhdGggZD0iTTkuOTM3IDE1LjVBMiAyIDAgMCAwIDguNSAxNC4wNjNsLTYuMTM1LTEuNTgyYS41LjUgMCAwIDEgMC0uOTYyTDguNSA5LjkzNkEyIDIgMCAwIDAgOS45MzcgOC41bDEuNTgyLTYuMTM1YS41LjUgMCAwIDEgLjk2MyAwTDE0LjA2MyA4LjVBMiAyIDAgMCAwIDE1LjUgOS45MzdsNi4xMzUgMS41ODJhLjUuNSAwIDAgMSAwIC45NjNMMTUuNSAxNC4wNjNhMiAyIDAgMCAwLTEuNDM3IDEuNDM3bC0xLjU4MiA2LjEzNWEuNS41IDAgMCAxLS45NjMgMHoiIC8+PHBhdGggZD0iTTIwIDN2NCIgLz48cGF0aCBkPSJNMjIgNWgtNCIgLz48L3N2Zz4=" class="gd-sparkle" />](skills.md)\
[llms.txt](llms.txt)\
[llms-full.txt](llms-full.txt)\


### Developers


**Uriah Finkel**


### Community

[Contributing guide](./contributing.html)\


### Meta

**Requires:** Python `>=3.9`\
[Package Info](package-info.md)

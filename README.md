# rtichoke

`rtichoke` is a Python library for visualizing the performance of predictive models. It provides a flexible and intuitive way to create a variety of common evaluation plots, including:

* **ROC Curves**
* **Precision-Recall Curves**
* **Gains and Lift Charts**
* **Calibration Curves**
* **Decision Curves**

The library is designed to be easy to use while still offering a high degree of control over the final plots.

For some reproducible examples please visit [rtichoke blog](https://rtichoke-blog.netlify.app/)!

## Installation

For a project managed with [uv](https://docs.astral.sh/uv/), add `rtichoke` with:

```bash
uv add rtichoke
```

Alternatively, install `rtichoke` from PyPI with pip:

```bash
pip install rtichoke
```

## Getting Started

To use `rtichoke`, you'll usually need two main inputs:

* `probs`: A dictionary containing model-predicted probabilities.
* `reals`: Observed outcomes, provided either as one array or as a dictionary keyed by population.

Here's a quick example of creating a ROC curve for a single model:

```python
import numpy as np
import rtichoke as rk

probs = {
    "Model A": np.array([0.1, 0.9, 0.4, 0.8, 0.3, 0.7, 0.2, 0.6])
}
reals = {
    "Population": np.array([0, 1, 0, 1, 0, 1, 0, 1])
}

fig = rk.create_roc_curve(
    probs=probs,
    reals=reals,
)

fig.show()
```

### Compare populations

When predictions and outcomes are both dictionaries with the same keys, rtichoke pairs them population-by-population. The populations do **not** need to have the same sample size.

```python
probs = {
    "Train": np.array([0.10, 0.90, 0.20, 0.80, 0.30, 0.70]),
    "Test": np.array([0.15, 0.85, 0.25, 0.75]),
}
reals = {
    "Train": np.array([0, 1, 0, 1, 0, 1]),
    "Test": np.array([0, 1, 0, 0]),
}

fig = rk.create_calibration_curve(
    probs=probs,
    reals=reals,
)

fig.show()
```

Here, `Train` contains six observations and `Test` contains four. Each probability vector only needs to match the outcome vector for its own population.

## Key Features

* **Simple API**: Create complex visualizations with a small amount of code.
* **Time-to-Event Analysis**: Support for time-dependent outcomes, including censoring and competing risks.
* **Interactive Plots**: Plotly-based interactive visualizations.
* **Flexible Data Handling**: Works with common Python array/data-frame workflows, including NumPy and Polars.

## Documentation

The official documentation, including the Getting Started guide and API reference, is published at:

https://uriahf.github.io/rtichoke_python/

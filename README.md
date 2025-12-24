# rtichoke

`rtichoke` is a Python library for visualizing the performance of predictive models. It provides a flexible and intuitive way to create a variety of common evaluation plots, including:

*   **ROC Curves**
*   **Precision-Recall Curves**
*   **Gains and Lift Charts**
*   **Decision Curves**

The library is designed to be easy to use, while still offering a high degree of control over the final plots. For some reproducible examples please visit the [rtichoke blog](https://uriahf.github.io/rtichoke-py/blog.html)!

## Installation

You can install `rtichoke` from PyPI:

```bash
pip install rtichoke
```

## Getting Started

To use `rtichoke`, you'll need two main inputs:

*   `probs`: A dictionary containing your model's predicted probabilities.
*   `reals`: A dictionary of the true binary outcomes.

Here's a quick example of how to create a ROC curve for a single model:

```python
import numpy as np
import rtichoke as rk

# Sample data for a model. Note that the probabilities for the
# positive class (1) are generally higher than for the negative class (0).
probs = {'Model A': np.array([0.1, 0.9, 0.4, 0.8, 0.3, 0.7, 0.2, 0.6])}
reals = {'Population': np.array([0, 1, 0, 1, 0, 1, 0, 1])}


# Create the ROC curve
fig = rk.create_roc_curve(
  probs=probs,
  reals=reals
)

fig.show()
```

## Key Features

*   **Simple API**: Create complex visualizations with just a few lines of code.
*   **Time-to-Event Analysis**: Native support for models with time-dependent outcomes, including censoring and competing risks.
*   **Interactive Plots**: Built on Plotly for interactive, publication-quality figures.
*   **Flexible Data Handling**: Works seamlessly with NumPy and Polars.

## Documentation

For a complete guide to the library, including a "Getting Started" tutorial and a full API reference, please see the **[official documentation](https://uriahf.github.io/rtichoke-py/)**.

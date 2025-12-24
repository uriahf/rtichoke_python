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

# For reproducibility
np.random.seed(42)

# Generate more realistic sample data for a model
# Probabilities for the positive class are generally higher
probs_positive_class = np.random.rand(50) * 0.5 + 0.5  # High probabilities (0.5 to 1.0)
probs_negative_class = np.random.rand(50) * 0.5       # Low probabilities (0.0 to 0.5)

# Combine and shuffle the data
probs_combined = np.concatenate([probs_positive_class, probs_negative_class])
reals_combined = np.concatenate([np.ones(50), np.zeros(50)])

shuffle_index = np.random.permutation(100)
probs = {'Model A': probs_combined[shuffle_index]}
reals = {'Population': reals_combined[shuffle_index]}


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

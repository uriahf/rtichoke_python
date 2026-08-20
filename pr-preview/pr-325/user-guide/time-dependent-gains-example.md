# Time-dependent gains example

This temporary example visually checks the perfect-model reference for multiple populations and multiple fixed time horizons.


``` python
import numpy as np
from rtichoke import create_gains_curve_times

probs = {
    "population_a": np.array([0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20, 0.10, 0.05]),
    "population_b": np.array([0.85, 0.75, 0.65, 0.55, 0.45, 0.35, 0.25, 0.15, 0.08, 0.03]),
}
reals = {
    "population_a": np.array([1, 1, 0, 1, 0, 0, 1, 0, 0, 0]),
    "population_b": np.array([1, 0, 1, 1, 1, 0, 1, 0, 0, 0]),
}
times = {
    "population_a": np.array([1.0, 2.0, 8.0, 3.0, 9.0, 10.0, 4.0, 7.0, 6.0, 11.0]),
    "population_b": np.array([1.0, 7.0, 2.0, 3.0, 8.0, 9.0, 4.0, 10.0, 6.0, 11.0]),
}

fig = create_gains_curve_times(
    probs=probs,
    reals=reals,
    times=times,
    fixed_time_horizons=[5.0, 10.0],
    by=0.05,
)
fig.show()
```


The rendered figure contains two populations evaluated at horizons 5 and 10. Each population-horizon combination should use its own horizon-specific event probability for the perfect-model reference, and the empirical gains curve should remain at or below that corresponding reference.

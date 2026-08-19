"""Generate the summary-report proof of concept used by PR previews."""

import numpy as np

from rtichoke import create_summary_report

rng = np.random.default_rng(2026)
n = 800
signal = rng.normal(size=n)
reals = rng.binomial(1, 1 / (1 + np.exp(-signal)))

probs = {
    "Model A": np.clip(1 / (1 + np.exp(-(0.9 * signal + rng.normal(0, 0.55, n)))), 0.001, 0.999),
    "Model B": np.clip(1 / (1 + np.exp(-(0.6 * signal + rng.normal(0, 0.85, n)))), 0.001, 0.999),
}

create_summary_report(probs, reals, output_file="summary-report-demo.html")

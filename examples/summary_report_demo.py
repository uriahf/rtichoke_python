"""Generate the summary-report proof of concept used by PR previews."""

import csv

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

# The PR preview renders the canonical R report from this exact dataset.  Using
# one serialized dataset avoids NumPy/R RNG differences obscuring visual parity.
with open("summary-report-reference-data.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["reals", "Model A", "Model B"])
    writer.writerows(zip(reals, probs["Model A"], probs["Model B"], strict=True))

create_summary_report(probs, reals, output_file="summary-report-demo.html")

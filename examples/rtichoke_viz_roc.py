"""Generate the rtichoke_viz browser-rendering proof from real rtichoke output."""

from pathlib import Path

import numpy as np

from rtichoke._viz_browser import _write_roc_browser_html
from rtichoke.performance_data.performance_data import prepare_performance_data

probs = {
    "Model A": np.array([0.05, 0.10, 0.20, 0.35, 0.55, 0.70, 0.85, 0.95]),
}
reals = np.array([0, 0, 0, 1, 0, 1, 1, 1])

performance_data = prepare_performance_data(probs=probs, reals=reals, by=0.1)
output = _write_roc_browser_html(
    performance_data,
    Path("rtichoke_viz_roc_proof") / "index.html",
)
print(output)

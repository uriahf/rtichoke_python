"""Generate a browser-rendered calibration proof from real rtichoke output."""

from pathlib import Path

import numpy as np

from rtichoke._viz_browser import _write_calibration_browser_html
from rtichoke.calibration.calibration import _create_calibration_curve_list

probs = {
    "Model A": np.array(
        [0.03, 0.08, 0.12, 0.18, 0.25, 0.32, 0.40, 0.50, 0.62, 0.75, 0.88, 0.96]
    )
}
reals = np.array([0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1])

calibration_curve_list = _create_calibration_curve_list(probs, reals)
output = _write_calibration_browser_html(
    calibration_curve_list,
    Path("rtichoke_viz_calibration_proof") / "index.html",
)
print(output)

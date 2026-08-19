"""Spike: export rtichoke's real Reactable performance table to standalone HTML.

This deliberately avoids Quarto/Jupyter as report assemblers. It uses the
ipywidgets static embed protocol and the existing rtichoke Reactable renderer.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from ipywidgets.embed import dependency_state, embed_data

from rtichoke.performance_data.performance_data import prepare_performance_data
from rtichoke.performance_table_reactable import render_performance_table_reactable


def main() -> None:
    reals = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    probs = {
        "Model A": np.array([0.05, 0.10, 0.15, 0.25, 0.35, 0.50, 0.60, 0.72, 0.82, 0.93]),
        "Model B": np.array([0.10, 0.20, 0.30, 0.35, 0.40, 0.45, 0.55, 0.65, 0.75, 0.85]),
    }
    performance_data = prepare_performance_data(
        probs=probs,
        reals=reals,
        stratified_by=("probability_threshold",),
        by=0.05,
    )
    table = render_performance_table_reactable(
        performance_data=performance_data,
        probs=probs,
        reals=reals,
        stratified_by="probability_threshold",
    )
    widget = table.to_widget()
    data = embed_data(views=[widget], state=dependency_state([widget]))

    html = f"""<!doctype html>
<html><head><meta charset='utf-8'><title>Reactable standalone spike</title>
<script src='https://cdnjs.cloudflare.com/ajax/libs/require.js/2.3.6/require.min.js'></script>
<script data-jupyter-widgets-cdn='https://cdn.jsdelivr.net/npm/' src='https://cdn.jsdelivr.net/npm/@jupyter-widgets/html-manager@*/dist/embed-amd.js'></script>
<script type='application/vnd.jupyter.widget-state+json'>{json.dumps(data['manager_state'])}</script>
</head><body>
<h1>rtichoke Reactable standalone spike</h1>
<p>No Quarto or running Jupyter kernel is used to view this page.</p>
<script type='application/vnd.jupyter.widget-view+json'>{json.dumps(data['view_specs'][0])}</script>
</body></html>"""
    out = Path("reactable-standalone-spike.html")
    out.write_text(html, encoding="utf-8")
    print(f"Wrote {out} ({out.stat().st_size / 1024:.1f} KiB HTML payload)")


if __name__ == "__main__":
    main()

"""Plotly-backed chart layer for the lightweight summary report.

The report shell, prevalence/AUROC widgets, and performance tables remain the
small self-contained HTML implementation.  Charts are rendered by Plotly—the
same rendering engine used by the canonical R summary report—so visual parity
is not limited by a hand-written SVG approximation.  Plotly.js is embedded
once in the generated file; no network access or new runtime dependency is
required.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Union

import numpy as np
from plotly.offline import get_plotlyjs

from rtichoke.calibration.calibration import create_calibration_curve
from rtichoke.discrimination.gains import plot_gains_curve
from rtichoke.discrimination.lift import plot_lift_curve
from rtichoke.discrimination.precision_recall import plot_precision_recall_curve
from rtichoke.discrimination.roc import plot_roc_curve
from rtichoke.performance_data.performance_data import prepare_performance_data
from rtichoke.summary_report.summary_report_v2 import (
    create_summary_report as _create_lightweight_report,
)
from rtichoke.utility.decision import plot_decision_curve


def _figure_payload(fig) -> dict:
    """Return JSON-safe Plotly data/layout without duplicating Plotly.js."""
    payload = json.loads(fig.to_json())
    return {"data": payload["data"], "layout": payload["layout"]}


def _plotly_payload(
    probs: Dict[str, np.ndarray],
    reals: Union[np.ndarray, Dict[str, np.ndarray]],
    by: float,
) -> dict:
    threshold = prepare_performance_data(
        probs=probs,
        reals=reals,
        stratified_by=("probability_threshold",),
        by=by,
    )
    ppcr = prepare_performance_data(
        probs=probs,
        reals=reals,
        stratified_by=("ppcr",),
        by=by,
    )

    def curves(data):
        return [
            {"label": "ROC", "figure": _figure_payload(plot_roc_curve(data, size=500))},
            {"label": "Lift", "figure": _figure_payload(plot_lift_curve(data, size=500))},
            {
                "label": "Precision Recall",
                "figure": _figure_payload(plot_precision_recall_curve(data, size=500)),
            },
            {"label": "Gains", "figure": _figure_payload(plot_gains_curve(data, size=500))},
        ]

    return {
        "smooth": _figure_payload(
            create_calibration_curve(
                probs=probs,
                reals=reals,
                calibration_type="smooth",
                size=550,
            )
        ),
        "discrete": _figure_payload(
            create_calibration_curve(
                probs=probs,
                reals=reals,
                calibration_type="discrete",
                size=550,
            )
        ),
        "threshold": curves(threshold),
        "ppcr": curves(ppcr),
        "decision": _figure_payload(plot_decision_curve(threshold, size=500)),
    }


def _inject_plotly_charts(html: str, payload: dict) -> str:
    plotly_js = get_plotlyjs()
    encoded = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    script = f"""
<script>{plotly_js}</script>
<script>
(function(){{
  const RP={encoded};
  const config={{displaylogo:false,responsive:false}};
  const draw=(id,fig)=>{{
    const host=document.getElementById(id); if(!host||!fig)return;
    Plotly.react(host,fig.data,fig.layout,config);
  }};

  // Replace the D3 calibration canvases with the same Plotly engine used by R.
  draw('smoothchart',RP.smooth);
  draw('discretechart',RP.discrete);
  draw('decision',RP.decision);

  function wireCurveTabs(specs,navId,chartId){{
    const nav=document.getElementById(navId), chart=document.getElementById(chartId);
    if(!nav||!chart)return;
    nav.replaceChildren();
    specs.forEach((spec,index)=>{{
      const button=document.createElement('button');
      button.textContent=spec.label;
      if(index===0)button.classList.add('active');
      button.addEventListener('click',()=>{{
        nav.querySelectorAll('button').forEach(b=>b.classList.toggle('active',b===button));
        Plotly.react(chart,spec.figure.data,spec.figure.layout,config);
      }});
      nav.appendChild(button);
    }});
    Plotly.react(chart,specs[0].figure.data,specs[0].figure.layout,config);
  }}
  wireCurveTabs(RP.threshold,'thrtabs','thrchart');
  wireCurveTabs(RP.ppcr,'pcrtabs','pcrchart');

  // Plotly calculates dimensions while a tab is visible. Resize when outer
  // R-Markdown-style tabs are activated so hidden PPCR/calibration plots do not
  // retain stale geometry.
  document.addEventListener('click',ev=>{{
    const b=ev.target.closest('button[data-target]'); if(!b)return;
    requestAnimationFrame(()=>{{
      ['smoothchart','discretechart','thrchart','pcrchart','decision'].forEach(id=>{{
        const node=document.getElementById(id);
        if(node&&node.offsetParent!==null)Plotly.Plots.resize(node);
      }});
    }});
  }});
}})();
</script>
"""
    return html.replace("</body>", script + "</body>", 1)


def create_summary_report(
    probs: Dict[str, np.ndarray],
    reals: Union[np.ndarray, Dict[str, np.ndarray]],
    output_file: str | Path = "summary_report.html",
    by: float = 0.01,
) -> Path:
    """Create the self-contained R-parity summary report using native Plotly charts."""
    out = Path(output_file)
    _create_lightweight_report(probs=probs, reals=reals, output_file=out, by=by)
    html = out.read_text(encoding="utf-8")
    html = _inject_plotly_charts(html, _plotly_payload(probs, reals, by))
    out.write_text(html, encoding="utf-8")
    return out

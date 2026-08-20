"""Plotly-backed chart layer for the lightweight summary report.

The report shell, prevalence/AUROC widgets, and performance tables remain the
small self-contained HTML implementation. Charts are rendered by Plotly—the
same rendering engine used by the canonical R summary report—so visual parity
is not limited by a hand-written SVG approximation. Plotly.js is embedded
once in the generated file; no network access or new runtime dependency is
required.
"""
from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Dict, Union

import numpy as np
from plotly.offline import get_plotlyjs

from rtichoke.calibration.calibration import create_calibration_curve
from rtichoke.performance_data.performance_data import prepare_performance_data
from rtichoke.processing.plotly_helper_functions import _plot_rtichoke_curve_binary
from rtichoke.summary_report.summary_report_v2 import (
    create_summary_report as _create_lightweight_report,
)


_PALETTE = [
    "#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#07004D", "#E6AB02",
    "#FE5F55", "#54494B", "#006E90", "#BC96E6", "#52050A", "#1F271B",
    "#BE7C4D", "#63768D", "#08A045", "#320A28", "#82FF9E", "#2176FF",
    "#D1603D", "#585123",
]


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

    def curves(data, stratified_by: str):
        return [
            {
                "label": "ROC",
                "figure": _figure_payload(
                    _plot_rtichoke_curve_binary(
                        data,
                        stratified_by=stratified_by,
                        curve="roc",
                        size=500,
                    )
                ),
            },
            {
                "label": "Lift",
                "figure": _figure_payload(
                    _plot_rtichoke_curve_binary(
                        data,
                        stratified_by=stratified_by,
                        curve="lift",
                        size=500,
                    )
                ),
            },
            {
                "label": "Precision Recall",
                "figure": _figure_payload(
                    _plot_rtichoke_curve_binary(
                        data,
                        stratified_by=stratified_by,
                        curve="precision recall",
                        size=500,
                    )
                ),
            },
            {
                "label": "Gains",
                "figure": _figure_payload(
                    _plot_rtichoke_curve_binary(
                        data,
                        stratified_by=stratified_by,
                        curve="gains",
                        size=500,
                    )
                ),
            },
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
        "threshold": curves(threshold, "probability_threshold"),
        "ppcr": curves(ppcr, "ppcr"),
        "decision": _figure_payload(
            _plot_rtichoke_curve_binary(
                threshold,
                stratified_by="probability_threshold",
                curve="decision",
                size=500,
            )
        ),
    }


def _wire_page_parity(
    html: str,
    probs: Dict[str, np.ndarray],
    reals: Union[np.ndarray, Dict[str, np.ndarray]],
) -> str:
    """Match the non-chart document flow of the canonical R Markdown report."""
    formulas = """<div class="metric-formulas r-math-blocks">
<div class="metric-equation">Prevalence = <span class="frac"><span>TP + FN</span><span>TP + FP + TN + FN</span></span></div>
<div class="metric-equation">PPCR (Predicted Positives Condition Rate) = <span class="frac"><span>TP + FP</span><span>TP + FP + TN + FN</span></span></div>
<div class="metric-equation">Sensitivity (Recall, True Positive Rate) = <span class="frac"><span>TP</span><span>TP + FN</span></span> = <span class="frac"><span>TP</span><span>Real Positives</span></span> = Prob( Predicted Positive | Real Positive )</div>
<div class="metric-equation">Specificity (True Negative Rate) = <span class="frac"><span>TN</span><span>TN + FP</span></span> = <span class="frac"><span>TN</span><span>Real Negatives</span></span> = Prob( Predicted Negative | Real Negative )</div>
<div class="metric-equation">PPV (Precision) = <span class="frac"><span>TP</span><span>TP + FP</span></span> = <span class="frac"><span>TP</span><span>Predicted Positives</span></span> = Prob( Real Positive | Predicted Positive )</div>
<div class="metric-equation">NPV = <span class="frac"><span>TN</span><span>TN + FN</span></span> = <span class="frac"><span>TN</span><span>Predicted Negatives</span></span> = Prob( Real Negative | Predicted Negative )</div>
<div class="metric-equation">Lift = <span class="frac"><span>PPV</span><span>Prevalence</span></span> = <span class="frac compound"><span><span class="frac compact"><span>TP</span><span>TP + FP</span></span></span><span><span class="frac compact"><span>TP + FN</span><span>TP + FP + TN + FN</span></span></span></span></div>
<div class="metric-equation">Net Benefit = <span class="frac"><span>TP</span><span>TP + FP + TN + FN</span></span> − <span class="frac"><span>FP</span><span>TP + FP + TN + FN</span></span> × <span class="frac"><span>p<sub>t</sub></span><span>1 − p<sub>t</sub></span></span></div>
</div></details>"""
    html, count = re.subn(
        r'<div class="metric-formulas">.*?</details>',
        formulas,
        html,
        count=1,
        flags=re.DOTALL,
    )
    if count != 1:
        raise RuntimeError("Could not locate summary-report metric formulas")

    parity_css = """
<style id="r-page-parity">
/* R Markdown leaves two explicit line breaks after the cheat sheet. */
details.metric-cheat-sheet, details:has(.r-math-blocks) { margin-bottom:40px; }
.r-math-blocks { margin:28px 0 18px; font-family:"STIXGeneral-Regular","Times New Roman",serif; font-size:16px; }
.r-math-blocks .metric-equation { display:block; margin:0 0 26px; min-height:28px; line-height:1.45; }
.r-math-blocks .frac { font-family:inherit; }
.r-math-blocks .frac.compound { font-size:.95em; vertical-align:middle; }
.r-math-blocks .frac.compact { margin:0; font-size:.92em; }
#prev.r-prevalence-multi { width:645px; max-width:100%; margin:28px 0 20px; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif; }
.r-prevalence-header,.r-prevalence-multi .prevalence-row { display:grid; grid-template-columns:45px 300px 300px; width:645px; max-width:100%; }
.r-prevalence-header { border-bottom:1px solid #ddd; font-weight:600; }
.r-prevalence-header span { padding:7px 10px; }
.r-prevalence-header span:first-child { padding:0; }
.r-prevalence-multi .prevalence-population,.r-prevalence-multi .prevalence-cell { grid-row:1; padding:7px 10px; }
.r-prevalence-multi .prevalence-population { display:flex; align-items:center; }
.r-prevalence-multi .prevalence-cell>strong { display:none; }
.r-prevalence-multi .prevalence-detail { grid-column:1/4; }
@media(max-width:700px){
  #prev.r-prevalence-multi,.r-prevalence-header,.r-prevalence-multi .prevalence-row{width:100%;grid-template-columns:36px minmax(120px,1fr) minmax(150px,1.2fr)}
}
</style>
"""
    html = html.replace("</head>", parity_css + "</head>", 1)

    if isinstance(reals, dict) and len(reals) > 1:
        sizes = {k: int(np.asarray(reals[k]).size) for k in probs if k in reals}
        sizes_json = json.dumps(sizes).replace("</", "<\\/")
        palette_json = json.dumps(_PALETTE)
        prevalence_script = f"""
<script>
(function(){{
  if(typeof SUM==='undefined'||SUM.length<2)return;
  const host=document.getElementById('prev'); if(!host)return;
  const sizes={sizes_json}, palette={palette_json};
  host.classList.add('r-prevalence-multi'); host.replaceChildren();
  const header=document.createElement('div'); header.className='r-prevalence-header';
  header.innerHTML='<span></span><span>population</span><span>Prevalence</span>'; host.appendChild(header);
  SUM.forEach((r,i)=>{{
    const p=Number(r.Prevalence), n=sizes[r.Model]||0, row=document.createElement('div'); row.className='prevalence-row';
    const exp=document.createElement('button'); exp.className='prevalence-expander'; exp.textContent='›'; exp.setAttribute('aria-label','Toggle details');
    const population=document.createElement('div'); population.className='prevalence-population';
    population.innerHTML='<span class="model-badge" style="background:'+palette[i%palette.length]+'"></span>'+r.Model;
    const cell=document.createElement('div'); cell.className='prevalence-cell';
    cell.innerHTML='<div class="prevalence-value"><span>'+p.toFixed(2)+'</span><span class="prevalence-track"><span style="width:'+Math.max(0,Math.min(100,p*100))+'%;background:grey"></span></span></div>';
    const detail=document.createElement('div'); detail.className='prevalence-detail'; detail.hidden=true;
    detail.textContent='Real Positives = '+Math.round(p*n)+',  Total Population =  '+n;
    exp.onclick=()=>{{detail.hidden=!detail.hidden;exp.textContent=detail.hidden?'›':'⌄'}};
    row.append(exp,population,cell,detail); host.appendChild(row);
  }});
}})();
</script>
"""
        html = html.replace("</body>", prevalence_script + "</body>", 1)

    return html


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
    html = _wire_page_parity(html, probs, reals)
    html = _inject_plotly_charts(html, _plotly_payload(probs, reals, by))
    out.write_text(html, encoding="utf-8")
    return out

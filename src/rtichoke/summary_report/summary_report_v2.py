"""Readable integration layer for the lightweight summary report renderers."""
from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Dict, Union

import numpy as np

from rtichoke.summary_report.curve_renderer import curve_renderer_source
from rtichoke.summary_report.summary_report import create_summary_report as _legacy_create_summary_report


def _asset_source(name: str) -> str:
    return Path(__file__).with_name(name).read_text(encoding="utf-8")


def _style_source() -> str:
    return _asset_source("report_style.css")


def _wire_curve_renderer(html: str) -> str:
    renderer = curve_renderer_source()
    marker = "function curveTabs(specs,nav,chart,strat)"
    if marker not in html:
        raise RuntimeError("Could not locate summary-report curve integration point")
    html = html.replace(marker, renderer + "\n" + marker, 1)
    html = html.replace("draw(s,chart,strat)}}));draw(specs[0],chart,strat)", "drawRtichokeCurve(s,chart,strat)}}));drawRtichokeCurve(specs[0],chart,strat)", 1)
    html = html.replace("draw(R.decision,'#decision','probability_threshold');", "drawRtichokeCurve(R.decision,'#decision','probability_threshold');", 1)
    return html


def _wire_performance_table_renderer(html: str) -> str:
    """Use the modular Reactable-like table renderer instead of legacy inline calls."""
    renderer = _asset_source("performance_table_renderer.js")
    marker = "perf(R.tables.threshold,'#table-threshold',false);perf(R.tables.ppcr,'#table-ppcr',true);"
    if marker not in html:
        raise RuntimeError("Could not locate summary-report performance-table integration point")
    return html.replace(marker, renderer, 1)


def _wire_report_style(html: str) -> str:
    """Replace the legacy inline CSS with the single parity stylesheet."""
    css = _style_source()
    styled, count = re.subn(r"<style>.*?</style>", f"<style>\n{css}\n</style>", html, count=1, flags=re.DOTALL)
    if count != 1:
        raise RuntimeError("Could not locate summary-report style element")
    return styled


def _wire_r_report_content(html: str, probs: Dict[str, np.ndarray], reals: Union[np.ndarray, Dict[str, np.ndarray]]) -> str:
    formulas = """<div class=\"metric-formulas\">
<div>Prevalence = <span class=\"frac\"><span>TP + FN</span><span>TP + FP + TN + FN</span></span></div>
<div>PPCR (Predicted Positives Condition Rate) = <span class=\"frac\"><span>TP + FP</span><span>TP + FP + TN + FN</span></span></div>
<div>Sensitivity (Recall, True Positive Rate) = <span class=\"frac\"><span>TP</span><span>TP + FN</span></span> = <span class=\"frac\"><span>TP</span><span>Real Positives</span></span> = Prob( Predicted Positive | Real Positive )</div>
<div>Specificity (True Negative Rate) = <span class=\"frac\"><span>TN</span><span>TN + FP</span></span> = <span class=\"frac\"><span>TN</span><span>Real Negatives</span></span> = Prob( Predicted Negative | Real Negative )</div>
<div>PPV (Precision) = <span class=\"frac\"><span>TP</span><span>TP + FP</span></span> = <span class=\"frac\"><span>TP</span><span>Predicted Positives</span></span> = Prob( Real Positive | Predicted Positive )</div>
<div>NPV = <span class=\"frac\"><span>TN</span><span>TN + FN</span></span> = <span class=\"frac\"><span>TN</span><span>Predicted Negatives</span></span> = Prob( Real Negative | Predicted Negative )</div>
<div>Lift = <span class=\"frac\"><span>PPV</span><span>Prevalence</span></span></div>
<div>Net Benefit = <span class=\"frac\"><span>TP</span><span>TP + FP + TN + FN</span></span> − <span class=\"frac\"><span>FP</span><span>TP + FP + TN + FN</span></span> × <span class=\"frac\"><span>p<sub>t</sub></span><span>1 − p<sub>t</sub></span></span></div>
</div>"""
    needle = "</table></div></details><div id=\"prev\"></div>"
    if needle in html:
        html = html.replace(needle, f"</table></div>{formulas}</details><div id=\"prev\"></div>", 1)

    if isinstance(reals, dict):
        sizes = {k: int(np.asarray(reals[k]).size) for k in probs if k in reals}
        prevalence_rows = "SUM"
    else:
        n = int(np.asarray(reals).size)
        sizes = {k: n for k in probs}
        prevalence_rows = "SUM.slice(0,1)"
    sizes_json = json.dumps(sizes).replace("</", "<\\/")
    script = f"""<script>
(function(){{
 const palette=['#1b9e77','#d95f02','#7570b3','#e7298a','#07004D','#E6AB02','#FE5F55','#54494B','#006E90','#BC96E6','#52050A','#1F271B','#BE7C4D','#63768D','#08A045','#320A28','#82FF9E','#2176FF','#D1603D','#585123'];
 const aucHost=document.getElementById('auc');
 if(aucHost&&typeof SUM!=='undefined'){{
   const showGroup=SUM.length>1;
   aucHost.innerHTML='<table class="summary-table auc-table"><thead><tr>'+(showGroup?'<th>Model</th>':'')+'<th>AUROC</th></tr></thead><tbody>'+SUM.map((r,i)=>{{
     const value=Number(r.AUC), valid=Number.isFinite(value), label=valid?value.toFixed(2):'    ', width=valid?Math.max(0,Math.min(100,value*100)):0;
     const group=showGroup?'<td><span class="model-badge" style="background:'+palette[i%palette.length]+'"></span>'+r.Model+'</td>':'';
     return '<tr>'+group+'<td><div class="prevalence-value"><span>'+label+'</span><span class="prevalence-track"><span style="width:'+width+'%;background:green"></span></span></div></td></tr>';
   }}).join('')+'</tbody></table>';
 }}
 const host=document.getElementById('prev'); if(!host||typeof SUM==='undefined')return; const sizes={sizes_json}; const prevalenceRows={prevalence_rows};
 host.innerHTML='';
 prevalenceRows.forEach((r,i)=>{{
   const p=Number(r.Prevalence), n=sizes[r.Model]||0, row=document.createElement('div'); row.className='prevalence-row';
   const exp=document.createElement('button'); exp.className='prevalence-expander'; exp.textContent='›'; exp.setAttribute('aria-label','Toggle details');
   const cell=document.createElement('div'); cell.className='prevalence-cell';
   cell.innerHTML='<strong>Prevalence</strong><div class="prevalence-value"><span>'+p.toFixed(2)+'</span><span class="prevalence-track"><span style="width:'+Math.max(0,Math.min(100,p*100))+'%"></span></span></div>';
   const detail=document.createElement('div'); detail.className='prevalence-detail'; detail.hidden=true;
   detail.textContent='Real Positives = '+Math.round(p*n)+',  Total Population =  '+n;
   exp.onclick=()=>{{detail.hidden=!detail.hidden;exp.textContent=detail.hidden?'›':'⌄'}};
   row.append(exp,cell,detail); host.append(row);
 }});
}})();
</script>"""
    return html.replace("</body>", script + "</body>", 1)


def create_summary_report(probs: Dict[str, np.ndarray], reals: Union[np.ndarray, Dict[str, np.ndarray]], output_file: str | Path = "summary_report.html", by: float = 0.01) -> Path:
    out = Path(output_file)
    _legacy_create_summary_report(probs=probs, reals=reals, output_file=out, by=by)
    html = out.read_text(encoding="utf-8")
    html = _wire_curve_renderer(html)
    html = _wire_performance_table_renderer(html)
    html = _wire_r_report_content(html, probs, reals)
    html = _wire_report_style(html)
    out.write_text(html, encoding="utf-8")
    return out

"""Lightweight HTML summary reports for rtichoke."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Union

import numpy as np

from rtichoke.performance_data.performance_data import prepare_performance_data


_CURVES = [
    ("roc", "ROC", "false_positive_rate", "sensitivity", "1 - Specificity", "Sensitivity"),
    ("precision-recall", "Precision–Recall", "sensitivity", "ppv", "Sensitivity", "PPV"),
    ("gains", "Gains", "ppcr", "sensitivity", "Predicted positives", "Sensitivity"),
    ("lift", "Lift", "ppcr", "lift", "Predicted positives", "Lift"),
    ("decision", "Decision Curve", "chosen_cutoff", "net_benefit", "Probability threshold", "Net benefit"),
]


def _json_rows(performance_data):
    columns = {
        "reference_group",
        "stratified_by",
        "chosen_cutoff",
        "false_positive_rate",
        "sensitivity",
        "ppv",
        "ppcr",
        "lift",
        "net_benefit",
    }
    available = [column for column in performance_data.columns if column in columns]
    data = performance_data.select(available)
    if "stratified_by" in available:
        data = data.filter(data["stratified_by"] == "probability_threshold")
    return data.to_dicts()


def _report_html(rows: list[dict]) -> str:
    payload = json.dumps(rows, separators=(",", ":"), default=str).replace("</", "<\\/")
    curves = json.dumps(_CURVES, ensure_ascii=False)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>rtichoke summary report</title>
<script src="https://cdn.jsdelivr.net/npm/d3@7/dist/d3.min.js"></script>
<style>
:root {{ color-scheme: light; font-family: Inter, ui-sans-serif, system-ui, sans-serif; color:#202124; }}
body {{ margin:0; background:#fafafa; }}
main {{ max-width:1180px; margin:auto; padding:42px 24px 64px; }}
h1 {{ margin:0 0 6px; font-size:2rem; }}
.lead {{ margin:0 0 28px; color:#62666d; }}
.toolbar {{ display:flex; flex-wrap:wrap; gap:10px; margin-bottom:22px; }}
.toggle {{ border:1px solid #d8dadd; border-radius:999px; background:white; padding:7px 12px; cursor:pointer; }}
.toggle.off {{ opacity:.35; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(360px,1fr)); gap:18px; }}
.card {{ background:white; border:1px solid #e2e3e5; border-radius:14px; padding:18px; box-shadow:0 1px 2px rgba(0,0,0,.03); }}
.card h2 {{ font-size:1.05rem; margin:0 0 8px; }}
svg {{ width:100%; height:auto; overflow:visible; }}
.axis text {{ fill:#686b70; font-size:11px; }} .axis path,.axis line {{ stroke:#d7d9dc; }}
.line {{ fill:none; stroke-width:2.2; }}
.ref {{ stroke:#a9adb3; stroke-dasharray:4 4; fill:none; }}
.tooltip {{ position:fixed; pointer-events:none; background:#202124; color:white; padding:7px 9px; border-radius:7px; font-size:12px; opacity:0; z-index:10; }}
.note {{ color:#777; font-size:.78rem; margin-top:24px; }}
</style>
</head>
<body><main>
<h1>Model Performance Summary</h1>
<p class="lead">One shared rtichoke performance-data calculation, rendered as lightweight linked charts.</p>
<div id="models" class="toolbar"></div><div id="charts" class="grid"></div>
<p class="note">D3 proof of concept. The preview currently loads D3 from a CDN; the production implementation will bundle it for a truly self-contained HTML file.</p>
</main><div class="tooltip"></div>
<script>
const data={payload}; const curves={curves};
const groups=[...new Set(data.map(d=>String(d.reference_group)))];
const active=new Set(groups); const color=d3.scaleOrdinal(groups,d3.schemeTableau10);
const toolbar=d3.select('#models');
groups.forEach(g=>{{ toolbar.append('button').attr('class','toggle').text(g).on('click',function(){{active.has(g)?active.delete(g):active.add(g);d3.select(this).classed('off',!active.has(g));drawAll();}}); }});
const tooltip=d3.select('.tooltip');
function finite(v){{return v!==null && v!==undefined && Number.isFinite(+v)}}
function drawAll(){{
 d3.select('#charts').selectAll('*').remove();
 curves.forEach(([id,title,xcol,ycol,xlabel,ylabel])=>{{
  const card=d3.select('#charts').append('section').attr('class','card'); card.append('h2').text(title);
  const width=500,height=330,m={{top:10,right:18,bottom:48,left:55}};
  const svg=card.append('svg').attr('viewBox',`0 0 ${{width}} ${{height}}`);
  let dd=data.filter(d=>active.has(String(d.reference_group))&&finite(d[xcol])&&finite(d[ycol]));
  let xDomain=(id==='roc'||id==='precision-recall'||id==='gains'||id==='lift')?[0,1]:d3.extent(dd,d=>+d[xcol]);
  if(!finite(xDomain[0])||xDomain[0]===xDomain[1]) xDomain=[0,1];
  let yDomain=(id==='roc'||id==='precision-recall'||id==='gains')?[0,1]:d3.extent(dd,d=>+d[ycol]);
  if(!finite(yDomain[0])||yDomain[0]===yDomain[1]) yDomain=[0,1];
  if(id==='decision') yDomain=[Math.min(0,yDomain[0]),Math.max(0,yDomain[1])];
  const x=d3.scaleLinear().domain(xDomain).nice().range([m.left,width-m.right]); const y=d3.scaleLinear().domain(yDomain).nice().range([height-m.bottom,m.top]);
  svg.append('g').attr('class','axis').attr('transform',`translate(0,${{height-m.bottom}})`).call(d3.axisBottom(x));
  svg.append('g').attr('class','axis').attr('transform',`translate(${{m.left}},0)`).call(d3.axisLeft(y));
  svg.append('text').attr('x',(m.left+width-m.right)/2).attr('y',height-8).attr('text-anchor','middle').attr('font-size',12).text(xlabel);
  svg.append('text').attr('transform','rotate(-90)').attr('x',-(m.top+height-m.bottom)/2).attr('y',14).attr('text-anchor','middle').attr('font-size',12).text(ylabel);
  if(id==='roc'||id==='gains') svg.append('path').attr('class','ref').attr('d',d3.line().x(d=>x(d[0])).y(d=>y(d[1]))([[0,0],[1,1]]));
  if(id==='decision') svg.append('line').attr('class','ref').attr('x1',x(x.domain()[0])).attr('x2',x(x.domain()[1])).attr('y1',y(0)).attr('y2',y(0));
  const line=d3.line().defined(d=>finite(d[xcol])&&finite(d[ycol])).x(d=>x(+d[xcol])).y(d=>y(+d[ycol]));
  groups.filter(g=>active.has(g)).forEach(g=>{{ const gd=dd.filter(d=>String(d.reference_group)===g).sort((a,b)=>+a[xcol]-+b[xcol]); svg.append('path').datum(gd).attr('class','line').attr('stroke',color(g)).attr('d',line); }});
  svg.append('rect').attr('x',m.left).attr('y',m.top).attr('width',width-m.left-m.right).attr('height',height-m.top-m.bottom).attr('fill','transparent').on('mousemove',event=>{{const [mx]=d3.pointer(event);const xv=x.invert(mx);let nearest=null,dist=Infinity;dd.forEach(d=>{{const z=Math.abs(+d[xcol]-xv);if(z<dist){{dist=z;nearest=d}}}});if(nearest) tooltip.style('opacity',1).style('left',(event.clientX+12)+'px').style('top',(event.clientY+12)+'px').html(`<b>${{nearest.reference_group}}</b><br>${{xlabel}}: ${{(+nearest[xcol]).toFixed(3)}}<br>${{ylabel}}: ${{(+nearest[ycol]).toFixed(3)}}`);}}).on('mouseleave',()=>tooltip.style('opacity',0));
 }});
}}
drawAll();
</script></body></html>"""


def create_summary_report(
    probs: Dict[str, np.ndarray],
    reals: Union[np.ndarray, Dict[str, np.ndarray]],
    output_file: str | Path = "summary_report.html",
    by: float = 0.01,
) -> Path:
    """Create a native HTML summary report for binary model performance.

    Performance data are prepared once and reused by all report panels.
    The current proof of concept renders ROC, precision-recall, gains, lift,
    and decision curves with D3.
    """
    performance_data = prepare_performance_data(
        probs=probs,
        reals=reals,
        stratified_by=("probability_threshold",),
        by=by,
    )
    output_path = Path(output_file)
    output_path.write_text(_report_html(_json_rows(performance_data)), encoding="utf-8")
    return output_path

"""Lightweight HTML summary reports for rtichoke."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Union

import numpy as np

from rtichoke.performance_data.performance_data import prepare_performance_data
from rtichoke.processing.plotly_helper_functions import _create_rtichoke_curve_list_binary


_CURVES = [
    ("roc", "ROC Curve"),
    ("precision recall", "Precision-Recall Curve"),
    ("gains", "Gains Curve"),
    ("lift", "Lift Curve"),
    ("decision", "Decision Curve"),
]


def _curve_specs(performance_data) -> list[dict]:
    """Build D3 specs from the same prepared curve data used by Plotly."""
    specs = []
    for curve, title in _CURVES:
        curve_data = _create_rtichoke_curve_list_binary(
            performance_data=performance_data,
            stratified_by="probability_threshold",
            curve=curve,
            size=600,
        )
        specs.append(
            {
                "id": curve.replace(" ", "-"),
                "title": title,
                "x_label": curve_data["x_label"],
                "y_label": curve_data["y_label"],
                "x_range": curve_data["axes_ranges"]["xaxis"],
                "y_range": curve_data["axes_ranges"]["yaxis"],
                "groups": curve_data["reference_group_keys"],
                "multiple_groups": curve_data["multiple_reference_groups"],
                "colors": curve_data["colors_dictionary"],
                "cutoffs": curve_data["cutoffs"],
                "data": curve_data["performance_data_ready_for_curve"].to_dicts(),
                "references": curve_data["reference_data"].to_dicts(),
            }
        )
    return specs


def _report_html(specs: list[dict]) -> str:
    payload = json.dumps(specs, separators=(",", ":"), default=str).replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>rtichoke summary report</title>
<script src="https://cdn.jsdelivr.net/npm/d3@7/dist/d3.min.js"></script>
<style>
:root {{ color-scheme:light; font-family:Arial,sans-serif; color:#2a3f5f; }}
body {{ margin:0; background:white; }}
main {{ max-width:1260px; margin:auto; padding:32px 24px 60px; }}
h1 {{ margin:0 0 28px; color:#2a3f5f; font-size:28px; font-weight:600; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(520px,1fr)); gap:34px 28px; }}
.card {{ min-width:0; }}
.card h2 {{ text-align:center; font-size:17px; font-weight:600; margin:0 0 2px; color:#2a3f5f; }}
.legend {{ min-height:28px; display:flex; justify-content:center; flex-wrap:wrap; gap:14px; font-size:12px; margin:5px 0 0; }}
.legend-item {{ display:inline-flex; align-items:center; gap:5px; cursor:pointer; user-select:none; }}
.legend-item.off {{ opacity:.3; }}
.legend-swatch {{ width:22px; height:3px; display:inline-block; }}
svg {{ width:100%; height:auto; overflow:visible; }}
.axis text {{ fill:#2a3f5f; font-size:12px; }}
.axis path,.axis line {{ stroke:#2a3f5f; stroke-width:1px; }}
.curve-line {{ fill:none; stroke-width:2px; }}
.curve-point {{ stroke:none; }}
.reference-line {{ fill:none; stroke-width:1.5px; stroke-dasharray:3 3; }}
.hover-marker {{ stroke:#000; stroke-width:3px; pointer-events:none; }}
.axis-label {{ fill:#2a3f5f; font-size:14px; }}
.slider-wrap {{ margin:2px 36px 0 62px; }}
.slider-label {{ font-size:13px; color:#2a3f5f; margin-bottom:2px; }}
.slider {{ width:100%; accent-color:#636efa; }}
.tooltip {{ position:fixed; pointer-events:none; padding:8px 10px; border-radius:2px; font-size:12px; line-height:1.35; opacity:0; z-index:10; color:white; box-shadow:0 1px 4px rgba(0,0,0,.18); }}
.note {{ margin-top:30px; color:#7f8da5; font-size:12px; }}
@media(max-width:620px) {{ .grid{{grid-template-columns:1fr}} main{{padding:22px 12px 50px}} }}
</style>
</head>
<body><main>
<h1>Model Performance Summary</h1>
<div id="charts" class="grid"></div>
<p class="note">D3 proof of concept using the same curve data, reference lines, axis ranges, palette, and hover content as rtichoke's Plotly figures.</p>
</main><div class="tooltip"></div>
<script>
const specs={payload};
const tooltip=d3.select('.tooltip');
const active=new Map();
specs.forEach(s=>s.groups.forEach(g=>{{if(!active.has(g)) active.set(g,true)}}));
function finite(v){{return v!==null&&v!==undefined&&Number.isFinite(+v)}}
function stripHtml(s){{return String(s??'').replace(/<br\\s*\\/?/gi,'\n').replace(/<\\/?b>/gi,'').replace(/<[^>]*>/g,'')}}
function htmlHover(s){{return String(s??'').replace(/NaN|nan/g,'')}}
function draw(spec){{
 const card=d3.select('#charts').append('section').attr('class','card');
 card.append('h2').text(spec.title);
 const legend=card.append('div').attr('class','legend');
 if(spec.multiple_groups) spec.groups.forEach(g=>{{
   const item=legend.append('span').attr('class','legend-item').classed('off',!active.get(g));
   item.append('span').attr('class','legend-swatch').style('background',spec.colors[g]); item.append('span').text(g);
   item.on('click',()=>{{active.set(g,!active.get(g)); redrawAll();}});
 }});
 const width=600,height=500,m={{top:22,right:28,bottom:62,left:72}};
 const svg=card.append('svg').attr('viewBox',`0 0 ${{width}} ${{height}}`);
 const x=d3.scaleLinear().domain(spec.x_range).range([m.left,width-m.right]);
 const y=d3.scaleLinear().domain(spec.y_range).range([height-m.bottom,m.top]);
 const xAxis=d3.axisBottom(x).ticks(6).tickSizeOuter(0); const yAxis=d3.axisLeft(y).ticks(6).tickSizeOuter(0);
 svg.append('g').attr('class','axis').attr('transform',`translate(0,${{height-m.bottom}})`).call(xAxis);
 svg.append('g').attr('class','axis').attr('transform',`translate(${{m.left}},0)`).call(yAxis);
 svg.append('text').attr('class','axis-label').attr('x',(m.left+width-m.right)/2).attr('y',height-15).attr('text-anchor','middle').text(spec.x_label);
 svg.append('text').attr('class','axis-label').attr('transform','rotate(-90)').attr('x',-(m.top+height-m.bottom)/2).attr('y',18).attr('text-anchor','middle').text(spec.y_label);
 const line=d3.line().defined(d=>finite(d.x)&&finite(d.y)).x(d=>x(+d.x)).y(d=>y(+d.y));
 const refs=d3.group(spec.references.filter(d=>finite(d.x)&&finite(d.y)),d=>String(d.reference_group));
 refs.forEach((rows,g)=>{{svg.append('path').datum(rows).attr('class','reference-line').attr('stroke',spec.colors[g]||'#BEBEBE').attr('d',line);}});
 const visible=spec.data.filter(d=>active.get(String(d.reference_group))&&finite(d.x)&&finite(d.y));
 spec.groups.filter(g=>active.get(g)).forEach(g=>{{
   const gd=visible.filter(d=>String(d.reference_group)===g);
   svg.append('path').datum(gd).attr('class','curve-line').attr('stroke',spec.colors[g]||'#000').attr('d',line);
   svg.selectAll(`.pt-${{spec.id}}-${{CSS.escape(g)}}`).data(gd).enter().append('circle').attr('class','curve-point').attr('cx',d=>x(+d.x)).attr('cy',d=>y(+d.y)).attr('r',2.3).attr('fill',spec.colors[g]||'#000');
 }});
 const markers=svg.append('g');
 function showCutoff(cutoff){{
   markers.selectAll('*').remove();
   spec.groups.filter(g=>active.get(g)).forEach(g=>{{
     const gd=visible.filter(d=>String(d.reference_group)===g&&finite(d.chosen_cutoff));
     if(!gd.length)return;
     const nearest=gd.reduce((a,b)=>Math.abs(+b.chosen_cutoff-cutoff)<Math.abs(+a.chosen_cutoff-cutoff)?b:a);
     markers.append('circle').attr('class','hover-marker').attr('cx',x(+nearest.x)).attr('cy',y(+nearest.y)).attr('r',6).attr('fill',spec.multiple_groups?(spec.colors[g]||'#000'):'#f6e3be');
   }});
 }}
 const overlay=svg.append('rect').attr('x',m.left).attr('y',m.top).attr('width',width-m.left-m.right).attr('height',height-m.top-m.bottom).attr('fill','transparent');
 overlay.on('mousemove',event=>{{
   const [mx,my]=d3.pointer(event); let nearest=null,dist=Infinity;
   visible.forEach(d=>{{const dx=x(+d.x)-mx,dy=y(+d.y)-my,z=dx*dx+dy*dy;if(z<dist){{dist=z;nearest=d}}}});
   if(nearest){{const c=spec.colors[String(nearest.reference_group)]||'#2a3f5f';tooltip.style('opacity',1).style('background',c).style('left',(event.clientX+12)+'px').style('top',(event.clientY+12)+'px').html(htmlHover(nearest.text));}}
 }}).on('mouseleave',()=>tooltip.style('opacity',0));
 if(spec.cutoffs.length){{
   const wrap=card.append('div').attr('class','slider-wrap'); const label=wrap.append('div').attr('class','slider-label');
   const slider=wrap.append('input').attr('class','slider').attr('type','range').attr('min',0).attr('max',spec.cutoffs.length-1).attr('step',1).property('value',0);
   const set=i=>{{const c=+spec.cutoffs[i];label.text(`Prob. Threshold: ${{Number.isFinite(c)?c.toFixed(2):c}}`);showCutoff(c)}};
   slider.on('input',function(){{set(+this.value)}}); set(0);
 }}
}}
function redrawAll(){{d3.select('#charts').selectAll('*').remove();specs.forEach(draw)}}
redrawAll();
</script></body></html>"""


def create_summary_report(
    probs: Dict[str, np.ndarray],
    reals: Union[np.ndarray, Dict[str, np.ndarray]],
    output_file: str | Path = "summary_report.html",
    by: float = 0.01,
) -> Path:
    """Create a native HTML summary report for binary model performance.

    Performance data are prepared once and reused by all report panels. The D3
    renderer consumes the same curve-ready data, reference lines, axis ranges,
    colors, and hover text as the existing Plotly implementation.
    """
    performance_data = prepare_performance_data(
        probs=probs,
        reals=reals,
        stratified_by=("probability_threshold",),
        by=by,
    )
    output_path = Path(output_file)
    output_path.write_text(_report_html(_curve_specs(performance_data)), encoding="utf-8")
    return output_path

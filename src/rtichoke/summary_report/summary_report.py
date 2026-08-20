"""Lightweight HTML summary reports for rtichoke."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Union
import numpy as np
from rtichoke.calibration.calibration import _create_calibration_curve_list
from rtichoke.performance_data.performance_data import prepare_performance_data
from rtichoke.processing.plotly_helper_functions import _create_rtichoke_curve_list_binary
from rtichoke.summary_report.calibration_renderer import calibration_renderer_source

_CURVES=[("roc","ROC"),("lift","Lift"),("precision recall","Precision Recall"),("gains","Gains")]
def _spec(data,strat,curve,label):
 d=_create_rtichoke_curve_list_binary(performance_data=data,stratified_by=strat,curve=curve,size=500)
 return {"label":label,"title":f"{label} Curve","x_label":d["x_label"],"y_label":d["y_label"],"x_range":d["axes_ranges"]["xaxis"],"y_range":d["axes_ranges"]["yaxis"],"groups":d["reference_group_keys"],"colors":d["colors_dictionary"],"data":d["performance_data_ready_for_curve"].to_dicts(),"references":d["reference_data"].to_dicts()}
def _specs(d,s): return [_spec(d,s,c,l) for c,l in _CURVES]
def _calibration(probs,reals):
 d=_create_calibration_curve_list(probs,reals,size=550); colors={k:v[0] for k,v in d["colors_dictionary"].items()}
 return {"deciles":d["deciles_dat"].to_dicts(),"smooth":d["smooth_dat"].to_dicts(),"reference":d["reference_data"].to_dicts(),"histogram":d["histogram_for_calibration"].to_dicts(),"ranges":d["axes_ranges"],"colors":colors,"groups":[k for k in colors if k!="reference_line"]}
def _auc(y,p):
 y=np.asarray(y).ravel().astype(int); p=np.asarray(p).ravel().astype(float); pos=p[y==1]; neg=p[y==0]
 return float("nan") if not len(pos) or not len(neg) else float(np.mean(pos[:,None]>neg[None,:])+.5*np.mean(pos[:,None]==neg[None,:]))
def _summaries(probs,reals):
 if isinstance(reals,dict) and probs.keys()==reals.keys(): return [{"Model":k,"Prevalence":float(np.mean(reals[k])),"AUC":_auc(reals[k],probs[k])} for k in probs]
 if not isinstance(reals,dict): return [{"Model":k,"Prevalence":float(np.mean(reals)),"AUC":_auc(reals,p)} for k,p in probs.items()]
 return []
def _table_data(d):
 cols=["reference_group","chosen_cutoff","ppcr","sensitivity","specificity","ppv","npv","lift","predicted_positives","net_benefit","true_posititives","true_negatives","false_positives","false_negatives"]
 return [{k:r.get(k) for k in cols if k in r} for r in d.to_dicts()]
def _html(payload,sums):
 P=json.dumps(payload,separators=(",",":"),default=str).replace("</","<\\/"); S=json.dumps(sums).replace("</","<\\/"); CAL=calibration_renderer_source(); D3=Path(__file__).with_name("microd3.js").read_text(encoding="utf-8")
 return f'''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Summary Report</title><script>{D3}</script><style>
*{{box-sizing:border-box}}body{{font-family:"Helvetica Neue",Helvetica,Arial,sans-serif;font-size:14px;line-height:1.42857143;color:#333;margin:0}}.main-container{{max-width:1040px;margin:auto;padding:20px 15px 60px}}h1{{font-size:36px;font-weight:500;line-height:1.1;margin:20px 0 10px}}h3{{font-size:24px;font-weight:500;margin:20px 0 10px}}#TOC{{margin:15px 0 28px}}#TOC ul{{margin:0;padding-left:20px}}#TOC>ul{{padding-left:0;list-style:none}}#TOC a{{color:#337ab7;text-decoration:none}}details{{margin:0 0 20px}}summary{{cursor:pointer}}summary p{{display:inline}}.cheat{{padding-top:15px;line-height:1.75}}table{{border-collapse:collapse;margin-bottom:20px}}th,td{{padding:7px 10px;text-align:center}}.good{{background:lightgreen;font-weight:600}}.bad{{background:pink;font-weight:600}}.summary-table{{min-width:310px;border:1px solid #eee}}.summary-table th,.summary-table td{{border-bottom:1px solid #eee}}.nav-tabs{{display:flex;flex-wrap:wrap;padding-left:0;margin:0 0 20px;list-style:none;border-bottom:1px solid #ddd}}.nav-tabs button{{color:#337ab7;background:transparent;border:1px solid transparent;border-radius:4px 4px 0 0;padding:10px 15px;margin-bottom:-1px;font:inherit;cursor:pointer}}.nav-tabs button.active{{color:#555;background:#fff;border-color:#ddd #ddd transparent}}.panel{{display:none}}.panel.active{{display:block}}.chart{{width:550px;max-width:100%;margin:0 auto 20px}}.plot-title,.legend,.axis text,.axis-label,.slider-label{{font-family:"Open Sans",verdana,arial,sans-serif;color:#444;fill:#444}}.plot-title{{text-align:center;font-size:17px;margin:0 0 2px}}.legend{{display:flex;justify-content:center;gap:16px;flex-wrap:wrap;font-size:12px;height:29px;align-items:center}}.legend span{{display:flex;align-items:center;gap:6px}}.legend i{{width:20px;height:2px;display:inline-block}}svg{{display:block;width:100%;height:auto}}.axis text{{font-size:12px}}.axis path,.axis line{{stroke:#444}}.grid line{{stroke:#eee}}.grid path{{display:none}}.line{{fill:none;stroke-width:2}}.ref{{fill:none;stroke-width:2;stroke-dasharray:3 3}}.axis-label{{font-size:14px}}.slider-wrap{{width:430px;max-width:calc(100% - 80px);margin:-54px auto 28px}}.slider-label{{font-size:16px}}input[type=range]{{width:100%}}.tip{{position:fixed;pointer-events:none;padding:8px 10px;background:#333;color:white;font:12px "Open Sans",sans-serif;opacity:0;z-index:20}}.perf-wrap{{overflow:auto;max-height:620px;border:1px solid #ddd;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}}.perf{{width:100%;margin:0;border-collapse:separate;border-spacing:0;font-size:14px}}.perf th{{position:sticky;top:0;background:#fff;z-index:2;white-space:nowrap;border-bottom:1px solid #ddd;font-weight:600;text-align:left;padding:8px 10px}}.perf .group-head th{{position:sticky;top:0;text-align:center;border-bottom:1px solid #ddd;font-weight:600}}.perf .column-head th{{top:35px}}.perf td{{border-bottom:1px solid #eee;text-align:left;padding:8px 10px;white-space:nowrap}}.perf tbody tr.data-row:hover{{background:#f5f5f5}}.perf .model{{text-align:left}}.model-badge{{display:inline-block;margin-right:8px;width:9px;height:9px;border-radius:50%;vertical-align:1px}}.metric-cell{{position:relative;isolation:isolate;min-width:72px}}.metric-cell::before{{content:"";position:absolute;z-index:-1;left:0;top:0;bottom:0;width:var(--bar,0%);background:var(--bar-color,lightgreen)}}.expand{{width:30px;cursor:pointer;font-size:18px;color:#777;text-align:center!important}}.detail td{{text-align:left;background:#fff;padding:16px}}.cm-title{{font-weight:600;margin-bottom:8px}}.cm{{display:inline-grid;grid-template-columns:auto auto;gap:3px;margin-left:10px}}.cm span{{padding:5px 10px;min-width:72px;text-align:center}}.pos{{background:lightgreen}}.neg{{background:pink}}
</style></head><body><div class="container-fluid main-container"><div id="header"><h1 class="title toc-ignore">Summary Report</h1></div><div id="TOC"><ul><li><a href="#calibration">Calibration</a></li><li><a href="#discrimination">Discrimination</a></li><li><a href="#utility-decision-curve">Utility (Decision Curve)</a></li><li><a href="#performance-table">Performance Table</a></li></ul></div>
<details><summary><p>Performance Metrics Cheat Sheet</p></summary><div class="cheat"><table><tr><th></th><th>Predicted Positive</th><th>Predicted Negative</th></tr><tr><th>Real Positive</th><td class="good">TP</td><td class="bad">FN</td></tr><tr><th>Real Negative</th><td class="bad">FP</td><td class="good">TN</td></tr></table></div></details><div id="prev"></div>
<div id="calibration"><h1>Calibration</h1><div class="nav-tabs"><button class="active" data-group="cal" data-target="smooth">Smooth</button><button data-group="cal" data-target="discrete">Discrete</button></div><div id="smooth" class="panel active"><div id="smoothchart" class="chart"></div></div><div id="discrete" class="panel"><div id="discretechart" class="chart"></div></div></div>
<div id="discrimination"><h1>Discrimination</h1><div id="auc"></div><div class="nav-tabs"><button class="active" data-group="disc" data-target="thr">By Probability Threshold</button><button data-group="disc" data-target="pcr">By Predicted Positives Condition Rate (PPCR)</button></div><div id="thr" class="panel active"><h3>Performance Metrics Curves</h3><div id="thrtabs" class="nav-tabs"></div><div id="thrchart" class="chart"></div></div><div id="pcr" class="panel"><h3>Performance Metrics Curves</h3><div id="pcrtabs" class="nav-tabs"></div><div id="pcrchart" class="chart"></div></div></div>
<div id="utility-decision-curve"><h1>Utility (Decision Curve)</h1><div id="decision" class="chart"></div></div><div id="performance-table"><h1>Performance Table</h1><div class="nav-tabs"><button class="active" data-group="tbl" data-target="t1">By Probability Threshold</button><button data-group="tbl" data-target="t2">By Predicted Positives Condition Rate (PPCR)</button></div><div id="t1" class="panel active"><div id="table-threshold"></div></div><div id="t2" class="panel"><div id="table-ppcr"></div></div></div></div><div class="tip"></div><script>
const R={P},SUM={S},tip=d3.select('.tip');
function summary(cols){{return `<table class="summary-table"><tr>${{cols.map(c=>`<th>${{c}}</th>`).join('')}}</tr>${{SUM.map(r=>`<tr>${{cols.map(c=>`<td>${{typeof r[c]==='number'?r[c].toFixed(3):r[c]}}</td>`).join('')}}</tr>`).join('')}}</table>`}}document.querySelector('#prev').innerHTML=summary(['Model','Prevalence']);document.querySelector('#auc').innerHTML=summary(['Model','AUC']);
document.addEventListener('click',e=>{{if(!e.target.matches('button[data-target]'))return;let g=e.target.dataset.group,t=e.target.dataset.target;document.querySelectorAll(`button[data-group="${{g}}"]`).forEach(b=>b.classList.toggle('active',b===e.target));(g==='cal'?['smooth','discrete']:g==='disc'?['thr','pcr']:['t1','t2']).forEach(id=>document.getElementById(id).classList.toggle('active',id===t))}});
function legend(c,s){{let l=c.append('div').attr('class','legend');s.groups.forEach(g=>l.append('span').html(`<i style="background:${{s.colors[g]||'#444'}}"></i>${{g}}`))}}
function draw(s,sel,strat){{let c=d3.select(sel);c.selectAll('*').remove();c.append('div').attr('class','plot-title').text(s.title);legend(c,s);let w=550,h=strat?675:550,m={{top:25,right:30,bottom:strat?145:70,left:60}},svg=c.append('svg').attr('viewBox',`0 0 ${{w}} ${{h}}`),x=d3.scaleLinear().domain(s.x_range).range([m.left,w-m.right]),y=d3.scaleLinear().domain(s.y_range).range([h-m.bottom,m.top]),line=d3.line().defined(d=>isFinite(+d.x)&&isFinite(+d.y)).x(d=>x(+d.x)).y(d=>y(+d.y));svg.append('g').attr('class','axis').attr('transform',`translate(0,${{h-m.bottom}})`).call(d3.axisBottom(x));svg.append('g').attr('class','axis').attr('transform',`translate(${{m.left}},0)`).call(d3.axisLeft(y));svg.append('text').attr('class','axis-label').attr('x',(m.left+w-m.right)/2).attr('y',h-m.bottom+45).attr('text-anchor','middle').text(s.x_label);svg.append('text').attr('class','axis-label').attr('transform','rotate(-90)').attr('x',-(m.top+h-m.bottom)/2).attr('y',18).attr('text-anchor','middle').text(s.y_label);d3.group(s.references,d=>String(d.reference_group)).forEach((a,g)=>svg.append('path').datum(a).attr('class','ref').attr('stroke',s.colors[g]||'#999').attr('d',line));s.groups.forEach(g=>svg.append('path').datum(s.data.filter(d=>String(d.reference_group)===g)).attr('class','line').attr('stroke',s.colors[g]||'#000').attr('d',line));}}
function curveTabs(specs,nav,chart,strat){{let n=d3.select(nav);specs.forEach((s,i)=>n.append('button').attr('class',i?'':'active').text(s.label).on('click',function(){{n.selectAll('button').classed('active',false);d3.select(this).classed('active',true);draw(s,chart,strat)}}));draw(specs[0],chart,strat)}}curveTabs(R.threshold,'#thrtabs','#thrchart','probability_threshold');curveTabs(R.ppcr,'#pcrtabs','#pcrchart','ppcr');draw(R.decision,'#decision','probability_threshold');
function calibration(type,sel){{
 const c=R.calibration,card=d3.select(sel); card.selectAll('*').remove();
 const W=550,H=550,X0=60,X1=540,MAIN_TOP=55,MAIN_BOTTOM=409.9,HIST_TOP=428.1,HIST_BOTTOM=510;
 const x=d3.scaleLinear().domain(c.ranges.xaxis).range([X0,X1]);
 const y=d3.scaleLinear().domain(c.ranges.yaxis).range([MAIN_BOTTOM,MAIN_TOP]);
 const histMax=d3.max(c.histogram,d=>+d.counts)||1;
 const yHist=d3.scaleLinear().domain([0,histMax]).nice().range([HIST_BOTTOM,HIST_TOP]);
 const svg=card.append('svg').attr('viewBox',`0 0 ${{W}} ${{H}}`);
 const defs=svg.append('defs'); defs.append('clipPath').attr('id',`main-${{type}}`).append('rect').attr('x',X0).attr('y',MAIN_TOP).attr('width',X1-X0).attr('height',MAIN_BOTTOM-MAIN_TOP); defs.append('clipPath').attr('id',`hist-${{type}}`).append('rect').attr('x',X0).attr('y',HIST_TOP).attr('width',X1-X0).attr('height',HIST_BOTTOM-HIST_TOP);
 if(c.groups.length>1){{const lg=svg.append('g').attr('font-family','Open Sans, verdana, arial, sans-serif').attr('font-size',12); const itemW=93,total=itemW*c.groups.length,start=300-total/2;c.groups.forEach((g,i)=>{{let q=lg.append('g').attr('transform',`translate(${{start+i*itemW}},24)`);q.append('line').attr('x1',5).attr('x2',35).attr('stroke',c.colors[g]).attr('stroke-width',2);q.append('text').attr('x',40).attr('y',4).attr('fill','#444').text(g)}})}}
 svg.append('g').attr('class','axis').attr('transform',`translate(${{X0}},0)`).call(d3.axisLeft(y));
 svg.append('g').attr('class','axis').attr('transform',`translate(${{X0}},0)`).call(d3.axisLeft(yHist).ticks(5));
 svg.append('g').attr('class','axis').attr('transform',`translate(0,${{HIST_BOTTOM}})`).call(d3.axisBottom(x).ticks(5));
 svg.append('text').attr('class','axis-label').attr('x',(X0+X1)/2).attr('y',548).attr('text-anchor','middle').text('Predicted');
 svg.append('text').attr('class','axis-label').attr('transform','rotate(-90)').attr('x',-(MAIN_TOP+MAIN_BOTTOM)/2).attr('y',18).attr('text-anchor','middle').text('Observed');
 const line=d3.line().defined(d=>isFinite(+d.x)&&isFinite(+d.y)).x(d=>x(+d.x)).y(d=>y(+d.y));
 const main=svg.append('g').attr('clip-path',`url(#main-${{type}})`);
 main.append('path').datum(c.reference).attr('fill','none').attr('stroke','#bebebe').attr('stroke-width',2).attr('stroke-dasharray','3,3').attr('d',line);
 const dat=type==='smooth'?c.smooth:c.deciles;
 c.groups.forEach(g=>{{let a=dat.filter(d=>String(d.reference_group)===g);main.append('path').datum(a).attr('fill','none').attr('stroke',c.colors[g]).attr('stroke-width',2).attr('d',line);if(type==='discrete')main.selectAll(null).data(a).enter().append('circle').attr('cx',d=>x(+d.x)).attr('cy',d=>y(+d.y)).attr('r',5).attr('fill',c.colors[g])}});
 const hist=svg.append('g').attr('clip-path',`url(#hist-${{type}})`),opacity=1/Math.max(1,c.groups.length);
 c.histogram.forEach(d=>{{let mid=+d.mids,left=x(mid-.005),right=x(mid+.005);hist.append('rect').attr('x',left).attr('width',Math.max(0,right-left)).attr('y',yHist(+d.counts)).attr('height',HIST_BOTTOM-yHist(+d.counts)).attr('fill',c.colors[String(d.reference_group)]||'#777').attr('opacity',opacity).attr('stroke','none').on('mousemove',ev=>tip.style('opacity',1).style('left',(ev.clientX+10)+'px').style('top',(ev.clientY+10)+'px').html(String(d.text||''))).on('mouseleave',()=>tip.style('opacity',0))}});
 const hoverData=dat.concat(c.reference);svg.append('rect').attr('x',X0).attr('y',MAIN_TOP).attr('width',X1-X0).attr('height',MAIN_BOTTOM-MAIN_TOP).attr('fill','transparent').on('mousemove',ev=>{{let [mx,my]=d3.pointer(ev),best=null,dist=Infinity;hoverData.forEach(d=>{{if(!isFinite(+d.x)||!isFinite(+d.y))return;let dd=(x(+d.x)-mx)**2+(y(+d.y)-my)**2;if(dd<dist){{dist=dd;best=d}}}});if(best&&dist<625)tip.style('opacity',1).style('left',(ev.clientX+10)+'px').style('top',(ev.clientY+10)+'px').html(String(best.text||''));else tip.style('opacity',0)}}).on('mouseleave',()=>tip.style('opacity',0));
}}
{CAL}
calibration('smooth','#smoothchart');calibration('discrete','#discretechart');
const LABEL={{reference_group:'Model',chosen_cutoff:'Probability Threshold',ppcr:'Predicted Positives',sensitivity:'Sens',specificity:'Spec',ppv:'PPV',npv:'NPV',lift:'Lift',predicted_positives:'Predicted Positives',net_benefit:'Net Benefit'}},MET=new Set(['sensitivity','specificity','ppv','npv']);function fmt(v){{return typeof v==='number'&&isFinite(v)?v.toFixed(2):(v??'')}}
function perf(rows,sel,isP){{
 const keys=(isP?['reference_group','ppcr']:['reference_group','chosen_cutoff']).concat(['sensitivity','specificity','ppv','npv','lift','net_benefit']);
 const maxLift=d3.max(rows,d=>Math.abs(+d.lift))||1,maxNB=d3.max(rows,d=>Math.abs(+d.net_benefit))||1;
 const groups=[...new Set(rows.map(r=>String(r.reference_group||'')))],palette=['#1b9e77','#d95f02','#7570b3','#e7298a','#07004D','#E6AB02'];
 const groupColor=new Map(groups.map((g,i)=>[g,palette[i%palette.length]]));
 let table=d3.select(sel).append('div').attr('class','perf-wrap').append('table').attr('class','perf');
 let gh=table.append('thead').append('tr').attr('class','group-head');gh.append('th').attr('colspan',isP?2:2);gh.append('th').attr('colspan',isP?5:6).text('Performance Metrics');
 let head=table.select('thead').append('tr').attr('class','column-head');head.append('th');keys.forEach(k=>head.append('th').text(LABEL[k]||k));
 let body=table.append('tbody');
 rows.forEach(r=>{{
   let tr=body.append('tr').attr('class','data-row'),ex=tr.append('td').attr('class','expand').text('›');
   keys.forEach(k=>{{
     let td=tr.append('td').attr('class',k==='reference_group'?'model':'');
     if(k==='reference_group'){{td.html(`<span class="model-badge" style="background:${{groupColor.get(String(r[k]||''))}}"></span>${{r[k]??''}}`);return}}
     if(k==='ppcr'){{const pct=isFinite(+r.ppcr)?Math.round(+r.ppcr*100):'';td.attr('class','metric-cell').style('--bar',`${{Math.max(0,Math.min(1,+r.ppcr||0))*100}}%`).style('--bar-color','lightgrey').text(`${{r.predicted_positives??''}} (${{pct}}%)`);return}}
     let width=0,color='lightgreen';if(MET.has(k))width=+r[k]||0;else if(k==='lift')width=(+r[k]||0)/maxLift;else if(k==='net_benefit'){{width=Math.abs(+r[k]||0)/maxNB;color=(+r[k]||0)<0?'pink':'lightgreen'}}
     if(MET.has(k)||k==='lift'||k==='net_benefit')td.attr('class','metric-cell').style('--bar',`${{Math.max(0,Math.min(1,width))*100}}%`).style('--bar-color',color);
     td.text(fmt(r[k]));
   }});
   let d=body.append('tr').attr('class','detail').style('display','none'),cell=d.append('td').attr('colspan',keys.length+1);
   cell.html(`<div class="cm-title">Confusion Matrix</div><span class="cm"><span class="pos">TP ${{fmt(r.true_positives)}}</span><span class="neg">FN ${{fmt(r.false_negatives)}}</span><span class="neg">FP ${{fmt(r.false_positives)}}</span><span class="pos">TN ${{fmt(r.true_negatives)}}</span></span>`);
   ex.on('click',()=>{{let open=d.style('display')!=='none';d.style('display',open?'none':'table-row');ex.text(open?'›':'⌄')}})
 }})
}}
perf(R.tables.threshold,'#table-threshold',false);perf(R.tables.ppcr,'#table-ppcr',true);
</script></body></html>'''
def create_summary_report(probs:Dict[str,np.ndarray],reals:Union[np.ndarray,Dict[str,np.ndarray]],output_file:str|Path="summary_report.html",by:float=.01)->Path:
 threshold=prepare_performance_data(probs=probs,reals=reals,stratified_by=("probability_threshold",),by=by); ppcr=prepare_performance_data(probs=probs,reals=reals,stratified_by=("ppcr",),by=by)
 payload={"threshold":_specs(threshold,"probability_threshold"),"ppcr":_specs(ppcr,"ppcr"),"decision":_spec(threshold,"probability_threshold","decision","Decision"),"calibration":_calibration(probs,reals),"tables":{"threshold":_table_data(threshold),"ppcr":_table_data(ppcr)}}
 out=Path(output_file); out.write_text(_html(payload,_summaries(probs,reals)),encoding="utf-8"); return out
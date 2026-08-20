/* R/Reactable-parity renderer for lightweight summary-report performance tables. */
(function () {
  const COLORS = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#07004D", "#E6AB02", "#FE5F55", "#54494B", "#006E90", "#BC96E6", "#52050A", "#1F271B", "#BE7C4D", "#63768D", "#08A045", "#320A28", "#82FF9E", "#2176FF", "#D1603D", "#585123"];
  const PAGE_SIZE = 10;
  const fmt = v => typeof v === "number" && isFinite(v) ? v.toFixed(2) : (v ?? "");
  const pct = v => typeof v === "number" && isFinite(v) ? `${(100 * v).toFixed(2)}%` : "";
  const num = v => typeof v === "number" && isFinite(v) ? v : 0;
  const esc = v => String(v ?? "").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#39;"}[c]));

  function injectStyles() {
    if (document.getElementById("rtichoke-perf-parity-css")) return;
    const style = document.createElement("style");
    style.id = "rtichoke-perf-parity-css";
    style.textContent = `
      .rt-perf-wrap{overflow:auto;border:1px solid #e5e5e5;border-radius:3px;background:#fff}
      .rt-perf{width:100%;border-collapse:separate;border-spacing:0;margin:0;font-size:14px}
      .rt-perf th,.rt-perf td{padding:8px 10px;text-align:left;border-bottom:1px solid #eee;white-space:nowrap;position:relative}
      .rt-perf thead th{background:#fff;font-weight:600;color:#333}
      .rt-perf .metric-group{text-align:center;border-bottom:1px solid #ddd}
      .rt-perf .model-dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:8px;vertical-align:1px}
      .rt-perf .expand{width:28px;text-align:center;color:#777;cursor:pointer;font-size:18px;padding-left:6px;padding-right:6px}
      .rt-perf .bar-cell{background-repeat:no-repeat;background-position:center;background-size:98% 88%}
      .rt-perf .detail td{background:#fafafa;padding:16px}
      .rt-conf{display:inline-table;border-collapse:collapse;margin:4px 0 4px 8px;vertical-align:middle}
      .rt-conf th,.rt-conf td{padding:6px 10px;border:1px solid #eee;text-align:left;min-width:105px}
      .rt-conf th{position:static;background:#fff;font-weight:600}
      .rt-conf .outcome{font-weight:600}
      .rt-pager{display:flex;align-items:center;justify-content:space-between;gap:16px;padding:8px 0;font-size:13px;color:#555}
      .rt-page-controls{display:flex;gap:4px;align-items:center}
      .rt-page-controls button{border:1px solid transparent;background:#fff;color:#337ab7;padding:5px 9px;border-radius:3px;font:inherit;cursor:pointer}
      .rt-page-controls button:hover:not(:disabled){background:#eee}
      .rt-page-controls button.active{background:#337ab7;color:#fff}
      .rt-page-controls button:disabled{color:#aaa;cursor:default}
    `;
    document.head.appendChild(style);
  }

  function metricBackground(value, maxValue=1, color="lightgreen") {
    if (!isFinite(+value) || maxValue <= 0) return "";
    const width = Math.min(Math.abs(+value) / maxValue, 1) * 100;
    return `linear-gradient(90deg, ${color} ${width}%, transparent ${width}%)`;
  }

  function nbBackground(value, maximum) {
    if (!isFinite(+value) || maximum <= 0) return "";
    const width = Math.max(-1, Math.min(+value / maximum, 1));
    const position = (0.5 + width / 2) * 100;
    return width >= 0
      ? `linear-gradient(90deg, transparent 50%, lightgreen 50%, lightgreen ${position}%, transparent ${position}%)`
      : `linear-gradient(90deg, transparent ${position}%, pink ${position}%, pink 50%, transparent 50%)`;
  }

  function confusionMatrix(r) {
    const tp=num(r.true_positives), tn=num(r.true_negatives), fp=num(r.false_positives), fn=num(r.false_negatives);
    const total=tp+tn+fp+fn || 1;
    const rows=[
      ["Predicted Positive",tp,fp,"lightgreen","pink"],
      ["Predicted Negative",fn,tn,"pink","lightgreen"],
      [" ",tp+fn,fp+tn,"lightgrey","lightgrey"]
    ];
    const value=(x)=>`${fmt(x)} (${(100*x/total).toFixed(2)}%)`;
    return `<table class="rt-conf"><thead><tr><th></th><th>Real Positive</th><th>Real Negative</th><th></th></tr></thead><tbody>${rows.map(q=>`<tr><td class="outcome">${q[0]}</td><td style="background:${metricBackground(q[1],total,q[3])}">${value(q[1])}</td><td style="background:${metricBackground(q[2],total,q[4])}">${value(q[2])}</td><td style="background:${metricBackground(q[1]+q[2],total,'lightgrey')}">${value(q[1]+q[2])}</td></tr>`).join("")}</tbody></table>`;
  }

  function render(rows, selector, isPpcr) {
    const host=document.querySelector(selector); if(!host) return;
    host.innerHTML="";
    const models=[...new Set(rows.map(r=>String(r.reference_group ?? "")))];
    const colors=Object.fromEntries(models.map((m,i)=>[m,COLORS[i%COLORS.length]]));
    const liftMax=Math.max(1e-12,...rows.map(r=>Math.abs(num(r.lift))));
    const nbMax=Math.max(1e-12,...rows.map(r=>Math.abs(num(r.net_benefit))));
    const sorted=[...rows].sort((a,b)=> isPpcr ? num(a.ppcr)-num(b.ppcr) : num(a.chosen_cutoff)-num(b.chosen_cutoff));
    const wrap=document.createElement("div"); wrap.className="rt-perf-wrap";
    const table=document.createElement("table"); table.className="rt-perf";
    const metricCount=isPpcr?5:6;
    table.innerHTML=`<thead><tr><th rowspan="2"></th><th rowspan="2">Model</th>${isPpcr?"":"<th rowspan=\"2\">Probability Threshold</th>"}<th rowspan="2">Predicted Positives</th><th class="metric-group" colspan="${metricCount}">Performance Metrics</th></tr><tr><th>Sens</th><th>Spec</th><th>PPV</th><th>NPV</th><th>Lift</th>${isPpcr?"":"<th>Net Benefit</th>"}</tr></thead>`;
    const body=document.createElement("tbody"); table.appendChild(body); wrap.appendChild(table); host.appendChild(wrap);
    const pager=document.createElement("div"); pager.className="rt-pager"; host.appendChild(pager);
    let page=0;

    function drawPage() {
      body.innerHTML="";
      const start=page*PAGE_SIZE, pageRows=sorted.slice(start,start+PAGE_SIZE);
      pageRows.forEach(r=>{
        const tr=document.createElement("tr");
        const model=String(r.reference_group ?? "");
        const ppcrText=`${fmt(r.predicted_positives)} (${pct(r.ppcr)})`;
        const metrics=[["sensitivity",1],["specificity",1],["ppv",1],["npv",1],["lift",liftMax]];
        tr.innerHTML=`<td class="expand">›</td><td><span class="model-dot" style="background:${colors[model]||'#aaa'}"></span>${esc(model)}</td>${isPpcr?"":`<td>${fmt(r.chosen_cutoff)}</td>`}<td class="bar-cell" style="background-image:${metricBackground(r.ppcr,1,'lightgrey')}">${ppcrText}</td>${metrics.map(([k,m])=>`<td class="bar-cell" style="background-image:${metricBackground(r[k],m)}">${fmt(r[k])}</td>`).join("")}${isPpcr?"":`<td class="bar-cell" style="background-image:${nbBackground(r.net_benefit,nbMax)}">${fmt(r.net_benefit)}</td>`}`;
        const detail=document.createElement("tr"); detail.className="detail"; detail.style.display="none";
        const td=document.createElement("td"); td.colSpan=isPpcr?9:10; td.innerHTML=confusionMatrix(r); detail.appendChild(td);
        tr.querySelector(".expand").addEventListener("click",e=>{const open=detail.style.display!=="none"; detail.style.display=open?"none":"table-row"; e.currentTarget.textContent=open?"›":"⌄";});
        body.appendChild(tr); body.appendChild(detail);
      });

      const pages=Math.ceil(sorted.length/PAGE_SIZE);
      if (pages <= 1) { pager.hidden=true; return; }
      pager.hidden=false; pager.innerHTML="";
      const info=document.createElement("span"); info.className="rt-page-info";
      info.textContent=`${start+1}–${Math.min(start+PAGE_SIZE,sorted.length)} of ${sorted.length} rows`;
      const controls=document.createElement("span"); controls.className="rt-page-controls";
      const button=(label,target,disabled,current=false)=>{const b=document.createElement("button");b.textContent=label;b.disabled=disabled;b.className=current?"active":"";b.addEventListener("click",()=>{page=target;drawPage();});return b;};
      controls.appendChild(button("Previous",Math.max(0,page-1),page===0));
      for(let i=0;i<pages;i++) controls.appendChild(button(String(i+1),i,false,i===page));
      controls.appendChild(button("Next",Math.min(pages-1,page+1),page===pages-1));
      pager.append(info,controls);
    }
    drawPage();
  }

  window.addEventListener("load",()=>{
    injectStyles();
    if (window.R && R.tables) {
      render(R.tables.threshold,"#table-threshold",false);
      render(R.tables.ppcr,"#table-ppcr",true);
    }
  });
})();

/* R/Reactable-parity renderer for lightweight summary-report performance tables. */
(function () {
  const COLORS = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#07004D", "#E6AB02", "#FE5F55", "#54494B", "#006E90", "#BC96E6", "#52050A", "#1F271B", "#BE7C4D", "#63768D", "#08A045", "#320A28", "#82FF9E", "#2176FF", "#D1603D", "#585123"];
  const PAGE_SIZE = 10;
  const fmt = v => typeof v === "number" && isFinite(v) ? v.toFixed(2) : (v ?? "");
  const pct = v => typeof v === "number" && isFinite(v) ? `${(100 * v).toFixed(2)}%` : "";
  const num = v => typeof v === "number" && isFinite(v) ? v : 0;
  const esc = v => String(v ?? "").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#39;"}[c]));

  if (!document.getElementById("rt-perf-control-style")) {
    const style=document.createElement("style");
    style.id="rt-perf-control-style";
    style.textContent=`
      .rt-check-inline{display:inline-flex!important;align-items:center;padding-left:0!important;margin-right:14px!important;gap:6px}
      .rt-check-inline input{position:absolute!important;opacity:0;pointer-events:none;margin:0!important}
      .rt-check-box{width:16px;height:16px;border:2px solid #333;border-radius:2px;display:grid;place-content:center;background:#fff;flex:0 0 auto}
      .rt-check-box>span{width:10px;height:10px;transform:scale(0);transition:transform .08s linear}
      .rt-dual-range{position:relative!important;height:50px!important;margin-top:2px!important}
      .rt-range-track{position:absolute;left:8px;right:8px;top:29px;height:6px;background:#e1e1e1;border-radius:3px}
      .rt-range-fill{position:absolute;top:0;height:6px;background:#337ab7;border-radius:3px}
      .rt-range-bubble{position:absolute;top:0;transform:translateX(-50%);padding:1px 5px;min-width:34px;text-align:center;background:#337ab7;color:#fff;border-radius:3px;font-size:11px;line-height:18px;white-space:nowrap}
      .rt-dual-range input[type=range]{-webkit-appearance:none;appearance:none;position:absolute!important;left:0!important;top:20px!important;width:100%!important;height:24px;margin:0!important;background:transparent!important;pointer-events:none!important;outline:none}
      .rt-dual-range input[type=range]::-webkit-slider-runnable-track{height:6px;background:transparent;border:0}
      .rt-dual-range input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;appearance:none;width:18px;height:18px;margin-top:-6px;border:1px solid #999;border-radius:50%;background:#fff;box-shadow:0 1px 2px rgba(0,0,0,.25);pointer-events:auto}
      .rt-dual-range input[type=range]::-moz-range-track{height:6px;background:transparent;border:0}
      .rt-dual-range input[type=range]::-moz-range-thumb{width:18px;height:18px;border:1px solid #999;border-radius:50%;background:#fff;box-shadow:0 1px 2px rgba(0,0,0,.25);pointer-events:auto}
      .rt-range-readout{display:none!important}
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
    const fp=num(r.false_positives), tn=num(r.true_negatives), fn=num(r.false_negatives);
    const tp=r.true_positives == null ? Math.max(0, num(r.predicted_positives)-fp) : num(r.true_positives);
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
    const valueOf=r=>isPpcr?num(r.ppcr):num(r.chosen_cutoff);
    const sorted=[...rows].sort((a,b)=>valueOf(a)-valueOf(b));
    const values=sorted.map(valueOf).filter(Number.isFinite);
    const minValue=values.length?Math.min(...values):0, maxValue=values.length?Math.max(...values):1;
    const step=Math.max(0.000001, ...values.slice(1).map((v,i)=>v-values[i]).filter(v=>v>0).slice(0,1), 0.01);
    const selected=new Set();
    let lower=minValue, upper=maxValue, page=0;

    // R crosstalk::bscols(widths = c(12, 6, 12)): group selector on a
    // full row, slider on a half-width row, then the full-width Reactable.
    const filters=document.createElement("div"); filters.className="rt-filters"; filters.style.display="block"; filters.style.marginBottom="15px";
    const modelFilter=document.createElement("div"); modelFilter.className="rt-filter-models"; modelFilter.style.width="100%"; modelFilter.style.marginBottom="15px";
    const modelLabel=document.createElement("div"); modelLabel.className="rt-filter-label"; modelLabel.textContent="Model"; modelFilter.appendChild(modelLabel);
    models.forEach((model,i)=>{
      const label=document.createElement("label"); label.className="rt-check-inline";
      const input=document.createElement("input"); input.type="checkbox"; input.value=model;
      const box=document.createElement("span"); box.className="rt-check-box";
      const boxFill=document.createElement("span"); boxFill.style.background=colors[model]||COLORS[i%COLORS.length]; box.appendChild(boxFill);
      const text=document.createElement("span"); text.textContent=model; label.append(input,box,text); modelFilter.appendChild(label);
      input.addEventListener("change",()=>{input.checked?selected.add(model):selected.delete(model);boxFill.style.transform=input.checked?"scale(1)":"scale(0)";page=0;drawPage();});
    });
    const rangeFilter=document.createElement("div"); rangeFilter.className="rt-filter-range"; rangeFilter.style.width="50%"; rangeFilter.style.maxWidth="520px"; rangeFilter.style.minWidth="300px";
    const rangeLabel=document.createElement("div"); rangeLabel.className="rt-filter-label"; rangeLabel.textContent=isPpcr?"Predicted Positives Condition Rate (PPCR)":"Probability Threshold";
    const rangeReadout=document.createElement("span"); rangeReadout.className="rt-range-readout";
    const track=document.createElement("div"); track.className="rt-dual-range";
    const rail=document.createElement("div"); rail.className="rt-range-track";
    const fill=document.createElement("div"); fill.className="rt-range-fill"; rail.appendChild(fill);
    const loBubble=document.createElement("span"), hiBubble=document.createElement("span"); loBubble.className=hiBubble.className="rt-range-bubble";
    const lo=document.createElement("input"), hi=document.createElement("input");
    [lo,hi].forEach(input=>{input.type="range";input.min=minValue;input.max=maxValue;input.step=step;}); lo.value=minValue; hi.value=maxValue;
    const sync=(redraw=true)=>{
      lower=Math.min(+lo.value,+hi.value);upper=Math.max(+lo.value,+hi.value);rangeReadout.textContent=`${fmt(lower)} – ${fmt(upper)}`;
      const span=maxValue-minValue||1, lp=100*(lower-minValue)/span, hp=100*(upper-minValue)/span;
      fill.style.left=`${lp}%`;fill.style.width=`${Math.max(0,hp-lp)}%`;
      loBubble.textContent=fmt(lower);hiBubble.textContent=fmt(upper);loBubble.style.left=`${lp}%`;hiBubble.style.left=`${hp}%`;
      if(redraw){page=0;drawPage();}
    };
    lo.addEventListener("input",sync); hi.addEventListener("input",sync); track.append(rail,loBubble,hiBubble,lo,hi); rangeFilter.append(rangeLabel,rangeReadout,track);
    if(models.length>1) filters.appendChild(modelFilter);
    filters.appendChild(rangeFilter); host.appendChild(filters);
    sync(false);

    const wrap=document.createElement("div"); wrap.className="rt-perf-wrap";
    const table=document.createElement("table"); table.className="rt-perf";
    if(isPpcr) {
      table.innerHTML='<thead><tr><th rowspan="2"></th><th rowspan="2">Model</th><th rowspan="2">Predicted Positives</th><th class="metric-group" colspan="5">Performance Metrics</th><th rowspan="2">Net Benefit</th></tr><tr><th>Sens</th><th>Spec</th><th>PPV</th><th>NPV</th><th>Lift</th></tr></thead>';
    } else {
      table.innerHTML='<thead><tr><th rowspan="2"></th><th rowspan="2">Probability Threshold</th><th rowspan="2">Model</th><th class="metric-group" colspan="6">Performance Metrics</th><th rowspan="2">Predicted Positives</th></tr><tr><th>Sens</th><th>Spec</th><th>PPV</th><th>NPV</th><th>Lift</th><th>Net Benefit</th></tr></thead>';
    }
    const body=document.createElement("tbody"); table.appendChild(body); wrap.appendChild(table); host.appendChild(wrap);
    const pager=document.createElement("div"); pager.className="rt-pager"; host.appendChild(pager);

    function filteredRows() {
      return sorted.filter(r=>{
        const model=String(r.reference_group ?? ""), value=valueOf(r);
        return (!selected.size||selected.has(model)) && value>=lower-1e-12 && value<=upper+1e-12;
      });
    }

    function drawPage() {
      const filtered=filteredRows();
      const pages=Math.max(1,Math.ceil(filtered.length/PAGE_SIZE)); if(page>=pages) page=pages-1;
      body.innerHTML="";
      const start=page*PAGE_SIZE, pageRows=filtered.slice(start,start+PAGE_SIZE);
      let previousThreshold=null;
      pageRows.forEach((r,rowIndex)=>{
        const tr=document.createElement("tr");
        const model=String(r.reference_group ?? "");
        const ppcrText=`${fmt(r.predicted_positives)} (${pct(r.ppcr)})`;
        const modelCell=`<td><span class="model-dot" style="background:${colors[model]||'#aaa'}"></span>${esc(model)}</td>`;
        const metrics=[["sensitivity",1],["specificity",1],["ppv",1],["npv",1],["lift",liftMax]];
        const metricCells=metrics.map(([k,m])=>`<td class="bar-cell" style="background-image:${metricBackground(r[k],m)}">${fmt(r[k])}</td>`).join("");
        const nbCell=`<td class="bar-cell" style="background-image:${nbBackground(r.net_benefit,nbMax)}">${fmt(r.net_benefit)}</td>`;
        const ppcrCell=`<td class="bar-cell" style="background-image:${metricBackground(r.ppcr,1,'lightgrey')}">${ppcrText}</td>`;
        if(isPpcr) {
          tr.innerHTML=`<td class="expand">›</td>${modelCell}${ppcrCell}${metricCells}${nbCell}`;
        } else {
          const threshold=fmt(r.chosen_cutoff);
          const repeated=rowIndex>0 && Math.abs(num(r.chosen_cutoff)-previousThreshold)<1e-12;
          const thresholdCell=`<td${repeated?' style="visibility:hidden"':''}>${threshold}</td>`;
          tr.innerHTML=`<td class="expand">›</td>${thresholdCell}${modelCell}${metricCells}${nbCell}${ppcrCell}`;
          previousThreshold=num(r.chosen_cutoff);
        }
        const detail=document.createElement("tr"); detail.className="detail"; detail.style.display="none";
        const td=document.createElement("td"); td.colSpan=isPpcr?9:10; td.innerHTML=confusionMatrix(r); detail.appendChild(td);
        tr.querySelector(".expand").addEventListener("click",e=>{const open=detail.style.display!=="none"; detail.style.display=open?"none":"table-row"; e.currentTarget.textContent=open?"›":"⌄";});
        body.appendChild(tr); body.appendChild(detail);
      });
      if (pages <= 1) { pager.hidden=true; return; }
      pager.hidden=false; pager.innerHTML="";
      const info=document.createElement("span"); info.className="rt-page-info";
      info.textContent=filtered.length?`${start+1}–${Math.min(start+PAGE_SIZE,filtered.length)} of ${filtered.length} rows`:`0 rows`;
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
    if (window.R && R.tables) {
      render(R.tables.threshold,"#table-threshold",false);
      render(R.tables.ppcr,"#table-ppcr",true);
    }
  });
})();

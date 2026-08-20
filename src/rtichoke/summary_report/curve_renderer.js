/* D3 renderer for performance and decision curves in the lightweight report.
 * Mirrors the R Plotly geometry used by create_summary_report(): a 500px-wide,
 * 550px-high widget with the animation slider inside the widget and no internal
 * title. ROC/Lift/etc. are supplied by the tab headings.
 */
function drawRtichokeCurve(s, sel, strat) {
  const card=d3.select(sel); card.selectAll("*").remove();
  card.style("width","500px").style("height","550px").style("max-width","100%").style("margin-left","0").style("margin-right","0").style("position","relative");
  const W=500,H=550,m={top:25,right:10,bottom:120,left:60};
  const svg=card.append("svg").style("width","500px").style("height","550px").style("max-width","100%").style("margin","0").attr("viewBox",`0 0 ${W} ${H}`),x=d3.scaleLinear().domain(s.x_range).range([m.left,W-m.right]),y=d3.scaleLinear().domain(s.y_range).range([H-m.bottom,m.top]);
  const line=d3.line().defined(d=>isFinite(+d.x)&&isFinite(+d.y)).x(d=>x(+d.x)).y(d=>y(+d.y));
  const styleAxis=a=>{a.attr("font-family","Open Sans, verdana, arial, sans-serif").attr("font-size",12).attr("color","#444");a.select(".domain").attr("stroke","#444");a.selectAll(".tick line").attr("stroke","#444")};
  const xa=svg.append("g").attr("class","axis").attr("transform",`translate(0,${H-m.bottom})`).call(d3.axisBottom(x).ticks(6)),ya=svg.append("g").attr("class","axis").attr("transform",`translate(${m.left},0)`).call(d3.axisLeft(y).ticks(6)); styleAxis(xa);styleAxis(ya);
  svg.append("text").attr("class","axis-label").attr("x",(m.left+W-m.right)/2).attr("y",H-m.bottom+42).attr("text-anchor","middle").text(s.x_label);
  svg.append("text").attr("class","axis-label").attr("transform","rotate(-90)").attr("x",-(m.top+H-m.bottom)/2).attr("y",18).attr("text-anchor","middle").text(s.y_label);
  const strategyColor=g=>{const k=String(g||"").toLowerCase();if(k==="treat_none")return "#808080";return s.colors[g]||"#BEBEBE"};
  const traces=[];
  d3.group(s.references,d=>String(d.reference_group)).forEach((a,g)=>{const color=strategyColor(g);svg.append("path").datum(a).attr("fill","none").attr("stroke",color).attr("stroke-width",2).attr("stroke-dasharray","2,4").attr("stroke-linecap","round").attr("d",line);traces.push(...a.map(d=>({...d,_color:color})))});
  const single=s.groups.length===1;
  s.groups.forEach(g=>{const a=s.data.filter(d=>String(d.reference_group)===g),color=single?"black":(s.colors[g]||"#000");svg.append("path").datum(a).attr("fill","none").attr("stroke",color).attr("stroke-width",2).attr("stroke-linejoin","round").attr("stroke-linecap","round").attr("d",line);svg.append("g").selectAll("circle").data(a.filter(d=>isFinite(+d.x)&&isFinite(+d.y))).enter().append("circle").attr("cx",d=>x(+d.x)).attr("cy",d=>y(+d.y)).attr("r",3).attr("fill",color).attr("stroke",color).attr("stroke-width",0);traces.push(...a.map(d=>({...d,_color:color})))});

  const markerLayer=svg.append("g").attr("class","curve-current-markers");
  const strataField=strat==="ppcr"?"ppcr":"chosen_cutoff";
  const strata=[...new Set(s.data.map(d=>+d[strataField]).filter(Number.isFinite))].sort((a,b)=>a-b);
  const drawCurrent=value=>{
    markerLayer.selectAll("*").remove();
    if(!strata.length)return;
    const nearest=strata.reduce((a,b)=>Math.abs(b-value)<Math.abs(a-value)?b:a,strata[0]);
    s.groups.forEach(g=>{
      const rows=s.data.filter(d=>String(d.reference_group)===g&&Number.isFinite(+d[strataField]));
      if(!rows.length)return;
      const d=rows.reduce((a,b)=>Math.abs(+b[strataField]-nearest)<Math.abs(+a[strataField]-nearest)?b:a,rows[0]);
      const color=single?"#f6e3be":(s.colors[g]||"#f6e3be");
      markerLayer.append("circle").datum({...d,_color:single?"black":(s.colors[g]||"#000")}).attr("cx",x(+d.x)).attr("cy",y(+d.y)).attr("r",6).attr("fill",color).attr("stroke","black").attr("stroke-width",3);
    });
  };

  if(strata.length>1){
    const wrap=card.append("div").attr("class","slider-wrap curve-slider-wrap").style("position","absolute").style("left","60px").style("bottom","12px").style("width","430px").style("max-width","calc(100% - 70px)").style("margin","0");
    const label=wrap.append("div").attr("class","slider-label curve-slider-label").style("font-size","12px").style("line-height","18px").style("color","black").style("margin","0 0 3px");
    const prefix=strat==="ppcr"?"Predicted Positives (Rate):":"Prob. Threshold:";
    const input=wrap.append("input").attr("class","curve-slider").attr("type","range").attr("min",strata[0]).attr("max",strata[strata.length-1]).attr("step",Math.max(1e-6,...strata.slice(1).map((v,i)=>v-strata[i]).filter(v=>v>0).slice(0,1))).style("width","100%").style("margin","0").style("accent-color","#777").node();
    input.value=strata[0];
    const update=()=>{const v=+input.value;label.textContent=`${prefix} ${Number.isFinite(v)?v.toFixed(2):""}`;drawCurrent(v)};
    input.addEventListener("input",update);update();
  }

  const contrast=color=>{const h=String(color||"#333").replace("#","");if(!/^[0-9a-f]{6}$/i.test(h))return "white";const r=parseInt(h.slice(0,2),16),g=parseInt(h.slice(2,4),16),b=parseInt(h.slice(4,6),16);return(.299*r+.587*g+.114*b)>170?"#222":"white"};
  const show=(ev,d)=>tip.style("opacity",1).style("left",(ev.clientX+10)+"px").style("top",(ev.clientY+10)+"px").style("background",d._color).style("border","1px solid "+d._color).style("color",contrast(d._color)).style("padding","6px 8px").style("border-radius","2px").style("font-family","Open Sans, verdana, arial, sans-serif").style("font-size","12px").style("line-height","15px").html(String(d.text||"")),hide=()=>tip.style("opacity",0);
  svg.append("rect").attr("x",m.left).attr("y",m.top).attr("width",W-m.left-m.right).attr("height",H-m.top-m.bottom).attr("fill","transparent").on("mousemove",ev=>{const[mx,my]=d3.pointer(ev);let best=null,dist=Infinity;traces.forEach(d=>{if(!isFinite(+d.x)||!isFinite(+d.y))return;const dd=(x(+d.x)-mx)**2+(y(+d.y)-my)**2;if(dd<dist){dist=dd;best=d}});if(best&&dist<225)show(ev,best);else hide()}).on("mouseleave",hide);
}

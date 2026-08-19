/* D3 renderer for performance and decision curves in the lightweight report.
 * Mirrors rtichoke's Plotly conventions: 600px figure, solid model traces,
 * dotted references, Plotly-like hover labels, and no grid lines.
 */
function drawRtichokeCurve(s, sel) {
  const card = d3.select(sel);
  card.selectAll("*").remove();
  card.append("div").attr("class", "plot-title").text(s.title);

  if (s.groups.length > 1) {
    const lg = card.append("div").attr("class", "legend");
    s.groups.forEach(g => lg.append("span").html(`<i style="background:${s.colors[g] || '#444'}"></i>${g}`));
  }

  const W = 600, H = 600, m = {top: 45, right: 35, bottom: 75, left: 75};
  const svg = card.append("svg").attr("viewBox", `0 0 ${W} ${H}`);
  const x = d3.scaleLinear().domain(s.x_range).range([m.left, W - m.right]);
  const y = d3.scaleLinear().domain(s.y_range).range([H - m.bottom, m.top]);
  const line = d3.line().defined(d => isFinite(+d.x) && isFinite(+d.y)).x(d => x(+d.x)).y(d => y(+d.y));

  const styleAxis = axis => {
    axis.attr("font-family", "Open Sans, verdana, arial, sans-serif").attr("font-size", 12).attr("color", "#444");
    axis.select(".domain").attr("stroke", "#444");
    axis.selectAll(".tick line").attr("stroke", "#444");
  };
  const xa = svg.append("g").attr("class", "axis").attr("transform", `translate(0,${H-m.bottom})`).call(d3.axisBottom(x).ticks(6));
  const ya = svg.append("g").attr("class", "axis").attr("transform", `translate(${m.left},0)`).call(d3.axisLeft(y).ticks(6));
  styleAxis(xa); styleAxis(ya);
  svg.append("text").attr("class", "axis-label").attr("x", (m.left + W-m.right)/2).attr("y", H-20).attr("text-anchor", "middle").text(s.x_label);
  svg.append("text").attr("class", "axis-label").attr("transform", "rotate(-90)").attr("x", -(m.top+H-m.bottom)/2).attr("y", 22).attr("text-anchor", "middle").text(s.y_label);

  const traces = [];
  d3.group(s.references, d => String(d.reference_group)).forEach((a,g) => {
    const color = s.colors[g] || "#999";
    svg.append("path").datum(a).attr("fill","none").attr("stroke",color).attr("stroke-width",2).attr("stroke-dasharray","3,3").attr("d",line);
    traces.push(...a.map(d => ({...d, _color: color})));
  });
  s.groups.forEach(g => {
    const a = s.data.filter(d => String(d.reference_group) === g);
    const color = s.colors[g] || "#000";
    svg.append("path").datum(a).attr("fill","none").attr("stroke",color).attr("stroke-width",2).attr("d",line);
    traces.push(...a.map(d => ({...d, _color: color})));
  });

  const contrast = color => {
    const h=String(color||"#333").replace("#",""); if(!/^[0-9a-f]{6}$/i.test(h)) return "white";
    const r=parseInt(h.slice(0,2),16),g=parseInt(h.slice(2,4),16),b=parseInt(h.slice(4,6),16);
    return (.299*r+.587*g+.114*b)>170 ? "#222" : "white";
  };
  const show = (ev,d) => tip.style("opacity",1).style("left",(ev.clientX+10)+"px").style("top",(ev.clientY+10)+"px")
    .style("background",d._color).style("border","1px solid "+d._color).style("color",contrast(d._color))
    .style("padding","6px 8px").style("border-radius","2px").html(String(d.text||""));
  const hide = () => tip.style("opacity",0);
  svg.append("rect").attr("x",m.left).attr("y",m.top).attr("width",W-m.left-m.right).attr("height",H-m.top-m.bottom).attr("fill","transparent")
    .on("mousemove", ev => {
      const [mx,my]=d3.pointer(ev); let best=null,dist=Infinity;
      traces.forEach(d=>{if(!isFinite(+d.x)||!isFinite(+d.y))return;const dd=(x(+d.x)-mx)**2+(y(+d.y)-my)**2;if(dd<dist){dist=dd;best=d}});
      if(best && dist<625) show(ev,best); else hide();
    }).on("mouseleave",hide);
}

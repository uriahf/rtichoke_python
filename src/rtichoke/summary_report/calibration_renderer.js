/* Calibration-only D3 renderer for the lightweight summary report.
 * Kept separate so visual parity with the R/Plotly report can be iterated
 * without touching the rest of the report template.
 */
function calibration(type, sel) {
  const c = R.calibration;
  const card = d3.select(sel);
  card.selectAll("*").remove();

  const W = 550, H = 550;
  const X0 = 60, X1 = 540;
  const MAIN_TOP = 55, MAIN_BOTTOM = 409.9;
  const HIST_TOP = 428.1, HIST_BOTTOM = 510;

  const x = d3.scaleLinear().domain(c.ranges.xaxis).range([X0, X1]);
  const y = d3.scaleLinear().domain(c.ranges.yaxis).range([MAIN_BOTTOM, MAIN_TOP]);
  const histMax = d3.max(c.histogram, d => +d.counts) || 1;
  const yHist = d3.scaleLinear().domain([0, histMax]).nice().range([HIST_BOTTOM, HIST_TOP]);
  const svg = card.append("svg").attr("viewBox", `0 0 ${W} ${H}`);

  const showTip = (ev, html, color) => {
    tip.style("opacity", 1)
      .style("left", (ev.clientX + 10) + "px")
      .style("top", (ev.clientY + 10) + "px")
      .style("background", color || "#333")
      .style("border", "1px solid " + (color || "#333"))
      .style("color", "white")
      .html(String(html || ""));
  };
  const hideTip = () => tip.style("opacity", 0);
  const groupOf = d => String(d.reference_group || "");
  const hoverColor = d => groupOf(d) === "reference_line" ? "#bebebe" : (c.colors[groupOf(d)] || "#777");

  const defs = svg.append("defs");
  defs.append("clipPath").attr("id", `main-${type}`).append("rect")
    .attr("x", X0).attr("y", MAIN_TOP)
    .attr("width", X1 - X0).attr("height", MAIN_BOTTOM - MAIN_TOP);
  defs.append("clipPath").attr("id", `hist-${type}`).append("rect")
    .attr("x", X0).attr("y", HIST_TOP)
    .attr("width", X1 - X0).attr("height", HIST_BOTTOM - HIST_TOP);

  if (c.groups.length > 1) {
    const lg = svg.append("g")
      .attr("font-family", "Open Sans, verdana, arial, sans-serif")
      .attr("font-size", 12);
    const itemW = 93, total = itemW * c.groups.length, start = 300 - total / 2;
    c.groups.forEach((g, i) => {
      const q = lg.append("g").attr("transform", `translate(${start + i * itemW},24)`);
      q.append("line").attr("x1", 5).attr("x2", 35)
        .attr("stroke", c.colors[g]).attr("stroke-width", 2);
      q.append("text").attr("x", 40).attr("y", 4).attr("fill", "#444").text(g);
    });
  }

  svg.append("g").attr("class", "axis").attr("transform", `translate(${X0},0)`).call(d3.axisLeft(y));
  svg.append("g").attr("class", "axis").attr("transform", `translate(${X0},0)`).call(d3.axisLeft(yHist).ticks(5));
  svg.append("g").attr("class", "axis").attr("transform", `translate(0,${HIST_BOTTOM})`).call(d3.axisBottom(x).ticks(5));
  svg.append("text").attr("class", "axis-label").attr("x", (X0 + X1) / 2)
    .attr("y", 548).attr("text-anchor", "middle").text("Predicted");
  svg.append("text").attr("class", "axis-label").attr("transform", "rotate(-90)")
    .attr("x", -(MAIN_TOP + MAIN_BOTTOM) / 2).attr("y", 18)
    .attr("text-anchor", "middle").text("Observed");

  const line = d3.line().defined(d => isFinite(+d.x) && isFinite(+d.y))
    .x(d => x(+d.x)).y(d => y(+d.y));
  const main = svg.append("g").attr("clip-path", `url(#main-${type})`);
  main.append("path").datum(c.reference).attr("fill", "none")
    .attr("stroke", "#bebebe").attr("stroke-width", 2)
    .attr("stroke-dasharray", "3,3").attr("d", line);

  const dat = type === "smooth" ? c.smooth : c.deciles;
  c.groups.forEach(g => {
    const a = dat.filter(d => String(d.reference_group) === g);
    main.append("path").datum(a).attr("fill", "none")
      .attr("stroke", c.colors[g]).attr("stroke-width", 2).attr("d", line);
    if (type === "discrete") {
      main.selectAll(null).data(a).enter().append("circle")
        .attr("cx", d => x(+d.x)).attr("cy", d => y(+d.y))
        .attr("r", 5).attr("fill", c.colors[g]);
    }
  });

  const hist = svg.append("g").attr("clip-path", `url(#hist-${type})`);
  const opacity = 1 / Math.max(1, c.groups.length);
  c.histogram.forEach(d => {
    const mid = +d.mids, left = x(mid - .005), right = x(mid + .005);
    hist.append("rect")
      .attr("x", left).attr("width", Math.max(0, right - left))
      .attr("y", yHist(+d.counts)).attr("height", HIST_BOTTOM - yHist(+d.counts))
      .attr("fill", c.colors[String(d.reference_group)] || "#777")
      .attr("opacity", opacity).attr("stroke", "none")
      .on("mousemove", ev => showTip(ev, d.text, hoverColor(d)))
      .on("mouseleave", hideTip);
  });

  const hoverData = dat.concat(c.reference);
  svg.append("rect").attr("x", X0).attr("y", MAIN_TOP)
    .attr("width", X1 - X0).attr("height", MAIN_BOTTOM - MAIN_TOP)
    .attr("fill", "transparent")
    .on("mousemove", ev => {
      const [mx, my] = d3.pointer(ev);
      let best = null, dist = Infinity;
      hoverData.forEach(d => {
        if (!isFinite(+d.x) || !isFinite(+d.y)) return;
        const dd = (x(+d.x) - mx) ** 2 + (y(+d.y) - my) ** 2;
        if (dd < dist) { dist = dd; best = d; }
      });
      if (best && dist < 625) showTip(ev, best.text, hoverColor(best));
      else hideTip();
    })
    .on("mouseleave", hideTip);
}

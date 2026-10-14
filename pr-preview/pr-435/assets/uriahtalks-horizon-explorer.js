// Great Docs-compatible renderer for the uriahtalks horizon explorer.
class UriahtalksHorizonExplorer extends HTMLElement {
  connectedCallback() {
    if (this.dataset.ready === "true") return;
    this.dataset.ready = "true";

    const competingAsCensored =
      this.hasAttribute("competing-as-censored");
    const min = Number(this.getAttribute("min") || 5);
    const max = Number(this.getAttribute("max") || 50);
    const step = Number(this.getAttribute("step") || 5);
    const initial = Number(this.getAttribute("horizon") || 30);
    const title = this.getAttribute("heading") || "Explore the horizon";
    const accent = this.getAttribute("accent-color") || "#7a9a01";

    this.style.setProperty("--uriah-horizon-accent", accent);
    this.innerHTML = `
      <div class="uriah-horizon-heading">${title}</div>
      <label class="uriah-horizon-control">
        <span>Fixed time horizon: <output>${initial}</output></span>
        <input type="range" min="${min}" max="${max}" step="${step}"
          value="${initial}" aria-label="Fixed time horizon">
      </label>
      <div class="uriah-horizon-chart" role="img"
        aria-label="Follow-up timelines classified at the selected horizon"></div>
      <div class="uriah-horizon-key"></div>
    `;

    const observations = [
      [1, .482, "primary"], [2, .194, "primary"],
      [3, .998, "primary"], [4, .372, "primary"],
      [5, .696, "none"], [6, .284, "competing"],
      [7, .784, "primary"], [8, .920, "competing"],
      [9, .630, "none"], [10, .086, "primary"],
    ].map(([id, fraction, outcome]) => ({
      id, outcome, time: fraction * max,
    }));

    const states = {
      primary: ["🤢", "Primary event"],
      competing: ["💀", "Competing event"],
      censored: ["🤬", "Censored"],
      nonEvent: ["🤨", "Non-event through horizon"],
    };
    const input = this.querySelector("input");
    const output = this.querySelector("output");
    const chart = this.querySelector(".uriah-horizon-chart");
    const key = this.querySelector(".uriah-horizon-key");
    const ns = "http://www.w3.org/2000/svg";

    const classify = (observation, horizon) => {
      if (observation.time > horizon) return states.nonEvent;
      if (observation.outcome === "primary") return states.primary;
      if (observation.outcome === "competing") {
        return competingAsCensored ? states.censored : states.competing;
      }
      return states.censored;
    };

    const render = () => {
      const horizon = Number(input.value);
      output.value = horizon;
      chart.replaceChildren();
      const width = 900, height = 390;
      const margin = { top: 22, right: 28, bottom: 48, left: 58 };
      const innerWidth = width - margin.left - margin.right;
      const rowHeight =
        (height - margin.top - margin.bottom) / observations.length;
      const x = value => margin.left + (value / max) * innerWidth;
      const svg = document.createElementNS(ns, "svg");
      svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
      svg.setAttribute("class", "uriah-horizon-svg");

      const tickStep = Math.max(step, max / 5);
      for (let tick = 0; tick <= max; tick += tickStep) {
        const line = document.createElementNS(ns, "line");
        line.setAttribute("x1", x(tick));
        line.setAttribute("x2", x(tick));
        line.setAttribute("y1", height - margin.bottom);
        line.setAttribute("y2", height - margin.bottom + 6);
        line.setAttribute("class", "uriah-axis");
        svg.appendChild(line);
        const text = document.createElementNS(ns, "text");
        text.setAttribute("x", x(tick));
        text.setAttribute("y", height - margin.bottom + 24);
        text.setAttribute("class", "uriah-tick");
        text.textContent = Number(tick.toFixed(2));
        svg.appendChild(text);
      }

      const axisLabel = document.createElementNS(ns, "text");
      axisLabel.setAttribute("x", margin.left + innerWidth / 2);
      axisLabel.setAttribute("y", height - 5);
      axisLabel.setAttribute("class", "uriah-axis-label");
      axisLabel.textContent = "Follow-up time";
      svg.appendChild(axisLabel);

      const horizonLine = document.createElementNS(ns, "line");
      horizonLine.setAttribute("x1", x(horizon));
      horizonLine.setAttribute("x2", x(horizon));
      horizonLine.setAttribute("y1", margin.top - 8);
      horizonLine.setAttribute("y2", height - margin.bottom);
      horizonLine.setAttribute("class", "uriah-horizon-line");
      svg.appendChild(horizonLine);

      observations.forEach((observation, index) => {
        const y = margin.top + rowHeight * (index + .5);
        const displayTime = Math.min(observation.time, horizon);
        const state = classify(observation, horizon);
        const label = document.createElementNS(ns, "text");
        label.setAttribute("x", margin.left - 14);
        label.setAttribute("y", y);
        label.setAttribute("class", "uriah-row-label");
        label.textContent = observation.id;
        svg.appendChild(label);
        const followup = document.createElementNS(ns, "line");
        followup.setAttribute("x1", x(0));
        followup.setAttribute("x2", x(displayTime));
        followup.setAttribute("y1", y);
        followup.setAttribute("y2", y);
        followup.setAttribute("class", "uriah-followup-line");
        svg.appendChild(followup);
        const marker = document.createElementNS(ns, "text");
        marker.setAttribute("x", x(displayTime));
        marker.setAttribute("y", y);
        marker.setAttribute("class", "uriah-emoji-marker");
        marker.textContent = state[0];
        const tooltip = document.createElementNS(ns, "title");
        tooltip.textContent =
          `Observation ${observation.id}: ${state[1]}; observed time ${observation.time.toFixed(2)}`;
        marker.appendChild(tooltip);
        svg.appendChild(marker);
      });

      chart.appendChild(svg);
      const active = competingAsCensored
        ? [states.primary, states.censored, states.nonEvent]
        : [states.primary, states.competing, states.censored, states.nonEvent];
      key.replaceChildren(...active.map(state => {
        const item = document.createElement("span");
        item.innerHTML = `<span aria-hidden="true">${state[0]}</span> ${state[1]}`;
        return item;
      }));
    };

    input.addEventListener("input", render);
    render();
  }
}

customElements.define("uriahtalks-horizon-explorer", UriahtalksHorizonExplorer);


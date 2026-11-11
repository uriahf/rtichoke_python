"""Real-browser checks of internally prepared terminal-event payloads."""

import copy
import json
from pathlib import Path
from typing import Any

import polars as pl
import pytest
from test_reals_distribution_steps_browser_acceptance import _serve

from rtichoke._renderers import RtichokeBrowserChart
from rtichoke._terminal_outcome_distribution import (
    _prepare_terminal_outcome_distributions,
    _terminal_outcome_distribution_v2_spec,
)


def test_terminal_outcome_status_browser_acceptance(tmp_path: Path):
    playwright = pytest.importorskip("playwright.sync_api")
    reference = json.loads(
        (Path(__file__).parent / "fixtures/terminal_event/reference.json").read_text()
    )

    def frame(name):
        return pl.DataFrame(reference["cases"][name]["data"]).rename(
            {"event": "reals", "time": "times"}
        )

    rows = _prepare_terminal_outcome_distributions(
        {"censored": frame("all_censored"), "zero ties": frame("zero_ties")},
        [0, 1, 2, 3],
    )
    spec = _terminal_outcome_distribution_v2_spec(rows)
    RtichokeBrowserChart(spec).write_html(tmp_path / "standalone.html")
    report: dict[str, Any] = {
        "schemaVersion": "1.1",
        "type": "report",
        "heuristics": {
            "active": "excluded:excluded",
            "available": list(spec["variants"]),
        },
        "sections": [
            {
                "id": "outcome-first",
                "title": "Outcome",
                "items": [{"type": "component", "id": "od", "spec": spec}],
            }
        ],
    }
    second = copy.deepcopy(report)
    second["sections"][0]["id"] = "outcome-second"
    # A test-only host exercises ReportSpec without wiring a public report API.
    (tmp_path / "reports.html").write_text(
        '<div id="first"></div><div id="second"></div><script type="module">'
        'import * as viz from "./rtichoke-viz.js";'
        f"const reports = {json.dumps([report, second], allow_nan=False)};"
        "reports.forEach((r, i) => {viz.assertReportReferentialIntegrity(r);"
        'document.querySelector(i ? "#second" : "#first").append(viz.renderReport(r));});'
        "</script>",
        encoding="utf-8",
    )
    with _serve(tmp_path) as base_url, playwright.sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(f"{base_url}/standalone.html")
        readout = ".rtichoke-outcome-distribution__readout"
        page.wait_for_selector(readout)
        assert "estimator plateaus" in page.locator(readout).inner_text()
        assert page.locator("svg").count() > 0
        page.goto(f"{base_url}/reports.html")
        page.wait_for_selector(f"#second {readout}")
        # Check every generated pair, population and horizon against its supplied
        # probabilities/accounting in both standalone and ReportSpec rendering.
        result = page.evaluate(
            """async ({spec, report}) => {
            const viz = await import('./rtichoke-viz.js');
            const check = (ok, message) => { if (!ok) throw Error(message); };
            const first = document.querySelector('#first');
            const second = document.querySelector('#second');
            const readout = root => root.querySelector('.rtichoke-outcome-distribution__readout');
            const choose = (root, selector) => {
              const radio = root.querySelector(selector);
              check(radio, 'missing selector: ' + selector);
              radio.checked = true;
              radio.dispatchEvent(new Event('change', {bubbles:true}));
            };
            const horizon = (root, index) => {
              const slider = root.querySelector('input[type="range"]');
              slider.value = String(index);
              slider.dispatchEvent(new Event('input', {bubbles:true}));
            };
            const initialSecond = readout(second).textContent;
            const verify = (root, dist, exclusion) => {
              const card = readout(root), text = card.textContent;
              check(text.includes('Time = ' + dist.horizon.toFixed(3)), 'wrong horizon');
              const counts = exclusion.excludedCounts.find(c => c.horizon === dist.horizon);
              for (const [heuristic, label, count] of [
                [dist.censoringHeuristic, 'Censored excluded', counts.censoredExcludedCount],
                [dist.competingHeuristic, 'Competing excluded', counts.competingExcludedCount]]) {
                if (heuristic === 'excluded') {
                  check(text.includes(label + count + ' / ' + exclusion.originalSampleSize + ' ('),
                    'lost exclusion: ' + JSON.stringify({count, exclusion, dist, text}));
                }
              }
              const stateRows = [...card.querySelectorAll('tbody tr')].slice(0, dist.states.length);
              check(stateRows.length === dist.states.length, 'incomplete state vector');
              dist.states.forEach((state, i) => {
                const row = stateRows[i];
                if (dist.estimateStatus === 'undefined_empty_population') {
                  check(row.textContent.includes('Unavailable'), 'null became available');
                  check(!row.querySelector('.rtichoke-performance-table__bar'), 'null has probability bar');
                } else {
                  check(row.textContent.includes((state.estimate * 100).toFixed(1) + '%'), 'probability changed');
                }
              });
              check(text.includes('no subjects remain') === (dist.estimateStatus === 'undefined_empty_population'), 'empty status lost');
              check(text.includes('estimator plateaus') === (dist.estimateStatus === 'beyond_support_plateau'), 'plateau status lost');
            };
            let contexts = 0;
            for (const [key, variant] of Object.entries(spec.variants)) {
              const [c, k] = key.split(':');
              choose(first, 'input[name$="-censoring"][value="' + c + '"]');
              choose(first, 'input[name$="-competing"][value="' + k + '"]');
              for (const evaluation of spec.evaluations) {
                choose(first, 'input[value="' + evaluation.id + '"]');
                const distributions = variant.stateDistributions.filter(d => d.evaluationId === evaluation.id);
                const exclusion = variant.exclusions.find(e => e.evaluationId === evaluation.id);
                const {variants, ...standaloneSpec} = spec;
                const standalone = viz.renderOutcomeDistribution({...standaloneSpec,
                  evaluations: [evaluation], stateDistributions: distributions, exclusions: [exclusion]});
                document.body.append(standalone);
                for (const [index, dist] of distributions.entries()) {
                  horizon(first, index); horizon(standalone, index);
                  verify(first, dist, exclusion); verify(standalone, dist, exclusion);
                  check(readout(second).textContent === initialSecond, 'report instance state leaked');
                  contexts++;
                }
                standalone.remove();
              }
              // A pair change must preserve an already selected horizon.
              horizon(first, 1);
              choose(first, 'input[name$="-censoring"][value="' + (c === 'excluded' ? 'adjusted' : 'excluded') + '"]');
              check(readout(first).textContent.includes('Time = 1.000'), 'heuristic change lost horizon');
            }
            const ids = [...document.querySelectorAll('[id]')].map(e => e.id);
            check(new Set(ids).size === ids.length, 'duplicate DOM identities');
            return contexts;
            }""",
            {"spec": spec, "report": report},
        )
        assert result == 8 * 2 * 4
        assert errors == []
        browser.close()

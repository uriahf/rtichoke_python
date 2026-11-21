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


def test_public_outcome_only_report_controls_and_offline_loading(tmp_path: Path):
    from rtichoke import create_reals_summary_report_times

    playwright = pytest.importorskip("playwright.sync_api")
    reference = json.loads(
        (Path(__file__).parent / "fixtures/terminal_event/reference.json").read_text()
    )
    populations = {
        name: reference["cases"][case]["data"]
        for name, case in [
            ("censored", "all_censored"),
            ("tied events", "tied_events_censor"),
        ]
    }
    reals = {name: [r["event"] for r in rows] for name, rows in populations.items()}
    times = {name: [r["time"] for r in rows] for name, rows in populations.items()}
    path = create_reals_summary_report_times(
        reals,
        times,
        [3, 1, 2, 1],
        heuristics=True,
        output_file=tmp_path / "public.html",
    )
    raw = create_reals_summary_report_times(
        reals,
        times,
        [1, 2, 3],
        output_file=tmp_path / "raw.html",
    )
    with _serve(tmp_path) as base_url, playwright.sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(f"{base_url}/{path.name}")
        page.wait_for_selector(".rtichoke-report__heuristics-controls")
        contexts = page.evaluate("""() => {
          const report = JSON.parse(document.querySelector('#rtichoke-report-spec').textContent);
          const spec = report.sections[0].items[0].spec;
          const root = document.querySelector('.rtichoke-report');
          const check = (ok, msg) => { if (!ok) throw Error(msg); };
          const choose = selector => {
            const radio = root.querySelector(selector);
            check(radio, 'missing control: ' + selector);
            radio.checked = true;
            radio.dispatchEvent(new Event('change', {bubbles:true}));
          };
          const readout = () => root.querySelector('.rtichoke-outcome-distribution__readout');
          let contexts = 0;
          for (const [key, variant] of Object.entries(spec.variants)) {
            const [c, k] = key.split(':');
            choose('input[name$="-censoring"][value="' + c + '"]');
            choose('input[name$="-competing"][value="' + k + '"]');
            for (const evaluation of spec.evaluations) {
              choose('input[value="' + evaluation.id + '"]');
              const exclusion = variant.exclusions.find(e => e.evaluationId === evaluation.id);
              const distributions = variant.stateDistributions.filter(d => d.evaluationId === evaluation.id);
              for (const [index, dist] of distributions.entries()) {
                const slider = root.querySelector('input[type="range"]');
                slider.value = String(index);
                slider.dispatchEvent(new Event('input', {bubbles:true}));
                const text = readout().textContent;
                check(text.includes('Time = ' + dist.horizon.toFixed(3)), 'lost horizon');
                const counts = exclusion.excludedCounts.find(e => e.horizon === dist.horizon);
                for (const [heuristic, label, count] of [
                  [c, 'Censored excluded', counts.censoredExcludedCount],
                  [k, 'Competing excluded', counts.competingExcludedCount]]) {
                  if (heuristic === 'excluded') {
                    check(text.includes(label + count + ' / ' + exclusion.originalSampleSize + ' ('), 'lost exclusion');
                  }
                }
                const rows = [...readout().querySelectorAll('tbody tr')].slice(0, dist.states.length);
                check(rows.length === dist.states.length, 'lost states');
                dist.states.forEach((state, i) => {
                  if (state.estimate === null) {
                    check(rows[i].textContent.includes('Unavailable'), 'undefined estimate became zero');
                    check(!rows[i].querySelector('.rtichoke-performance-table__bar'), 'undefined estimate has bar');
                  } else {
                    check(rows[i].textContent.includes((state.estimate * 100).toFixed(1) + '%'), 'estimate changed');
                  }
                });
                check(text.includes('no subjects remain') === (dist.estimateStatus === 'undefined_empty_population'), 'lost undefined notice');
                check(text.includes('estimator plateaus') === (dist.estimateStatus === 'beyond_support_plateau'), 'lost support notice');
                contexts++;
              }
            }
            choose('input[name$="-censoring"][value="' + (c === 'excluded' ? 'adjusted' : 'excluded') + '"]');
            check(readout().textContent.includes('Time = 3.000'), 'pair change lost horizon');
            check(root.querySelector('input[value="evaluation-2"]').checked, 'pair change lost population');
          }
          const ids = [...document.querySelectorAll('[id]')].map(e => e.id);
          check(new Set(ids).size === ids.length, 'duplicate DOM ids');
          return contexts;
        }""")
        assert contexts == 64
        note = page.locator(".rtichoke-cheat-sheet").text_content()
        assert "independent-censoring assumptions" in note
        assert "unavailable estimates, not zero" in note
        page.goto(f"{base_url}/{raw.name}")
        page.wait_for_selector(".rtichoke-outcome-distribution__readout")
        assert page.locator(".rtichoke-report__heuristics-controls").count() == 0
        assert (
            "raw observed outcome status"
            in page.locator(".rtichoke-cheat-sheet").text_content()
        )
        requests = []
        page.on("request", lambda request: requests.append(request.url))
        page.goto(path.as_uri())
        page.wait_for_selector(".rtichoke-report__heuristics-controls")
        assert not any(url.startswith(("http:", "https:")) for url in requests)
        assert errors == []
        browser.close()

# rtichoke Python repository rules

## Role and ownership

This repository owns the Python implementation of rtichoke: Python statistical
calculations and public APIs, performance-data preparation, evaluation and
population metadata, canonical visualization adapters, immutable
`rtichoke_viz` vendoring, browser-renderer consumer adoption, and Python tests,
typing, linting, packaging, and documentation.

It does not own shared TypeScript visualization contracts or renderers.

## Start from fresh state

Before modifying anything:

1. Inspect actual current `main`, relevant open PRs, recent relevant merges, and
   tags or releases when relevant.
2. Check whether equivalent work already exists.
3. Inspect the repository's current scripts, tests, workflows, and conventions
   instead of relying on remembered commands or architecture.

If actual repository state materially contradicts the task assumptions, stop
before broadening scope and report the discrepancy.

## Scope discipline

Make the smallest change required. During focused work, do not opportunistically
redesign unrelated APIs, change statistics outside scope, modify
`rtichoke_viz` source, expand summary-report behavior, introduce unrelated
dependencies, or perform broad refactoring. Stop and report when a task requires
a materially broader architectural or statistical decision.

Preserve existing public APIs and defaults unless explicitly requested. Static
behavior must remain unchanged when adding time-dependent browser adoption. Do
not add R-like APIs merely for cross-language symmetry, or add dependencies or
breaking behavior without explicit scope.

## Python conventions

- Prefer Polars for production dataframe work and `uv` for environment and
  dependency workflows.
- Prefer existing project patterns over new abstractions and a functional style
  where practical.
- Do not introduce pandas as a direct production dependency unless the task
  explicitly requires it.
- Give exported functions accurate NumPy-style docstrings.
- Keep Ruff formatting and linting and `ty` type checking green.

## Statistical boundary

Existing production statistical calculations are authoritative unless a task
explicitly targets a statistical bug or methodology change. Browser and
canonical-visualization adoption must consume already-computed production
quantities rather than recomputing statistics in adapters.

Do not change censoring logic, Aalen-Johansen logic, competing-risk handling, or
cutoff semantics unless explicitly requested. Preserve established static and
time-dependent cutoff behavior where they intentionally differ.

## Evaluation and reference identity

Preserve semantic identity:

- Evaluation identity represents the semantic model/population evaluation.
- Rendered-series geometry identity is distinct from evaluation identity.
- Horizon is evaluation-context and geometry metadata, not a replacement for
  evaluation identity.
- Reuse `_EvaluationMetadata` and related semantic infrastructure where
  appropriate.
- Never infer population identity from numerical equality. Distinct populations
  remain distinct even when prevalence, event risk, or reference geometry is
  numerically equal.

Use explicit semantic reference ownership. Scope population-dependent references
by semantic population, and by `population × horizon` for time-dependent outputs
when required. Multiple models sharing one population must share the same
population-owned reference where semantics require it. Do not derive ownership
from equal or unequal numeric values.

## Canonical visualization and renderers

Canonical adapters map already-computed Python output into the shared contract.
They must preserve semantic evaluation identity and reference ownership, create
deterministic series geometry identity, carry horizon metadata where applicable,
and avoid statistical recomputation. Extend the existing adapter architecture
instead of creating parallel one-off implementations.

Plotly remains the default renderer unless a task explicitly changes public
defaults. Canonical browser rendering remains opt-in unless explicitly changed.
Reuse the established public renderer vocabulary and `RtichokeBrowserChart`
infrastructure. Never silently change default output types.

## Immutable `rtichoke_viz` consumption

Never consume `rtichoke_viz/main`; use only immutable, verified releases and the
repository's established vendoring mechanism. Before vendoring a release,
verify all of the following:

- tag and exact source commit;
- archive filename and SHA-256 against the published checksum;
- `MANIFEST` version and source commit;
- required packaged JavaScript, CSS, and schema files;
- required public renderer exports.

Never manually edit vendored renderer bytes. On upgrade, search the whole
repository for stale version, tag, archive, and checksum pins, then synchronize
all provenance records, tests, packaging assertions, and workflow guards.

## Validation

Before opening a PR, inspect the actual current scripts and workflows, run
focused tests and the complete relevant validation suite, and do not weaken
tests merely to make CI green. Run real-browser acceptance when browser rendering
changes, and packaging and provenance guards when vendored assets change.

Current baseline commands include:

```bash
uv sync --all-extras --dev
uv run ruff check .
uv run ruff format --check .
uv run ty check src/rtichoke
uv run pytest tests
```

Use the repository's actual current commands when they differ or when more
focused validation is appropriate.

## PR and release ownership

For mutation tasks:

1. Implement the focused change and validate it locally.
2. Open one focused PR.
3. Inspect GitHub Actions for the current PR head and, while the session is
   active, recheck jobs that are still running.
4. Inspect failed job logs, fix in-scope failures, push, and repeat until green
   or genuinely blocked.

Do not ask the user to manually check CI. Escalate only when resolution requires
a broader semantic, statistical, compatibility, dependency, product, or
infrastructure decision. Do not merge unless explicitly instructed.

Do not publish Python package releases unless explicitly requested. Consumer
adoption normally stops at an unmerged focused PR.

## Completion report

At the end of mutation work, report:

- starting `main`, final branch and head, and package version;
- files changed;
- statistical behavior changed or explicitly unchanged;
- canonical identity and reference behavior;
- vendored visualization provenance when applicable;
- local validation and final CI status;
- PR number, link, and state;
- anything deliberately deferred.

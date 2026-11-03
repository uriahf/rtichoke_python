# Independent R multistate reference

Run `Rscript tests/fixtures/multistate_aj/generate_r_multistate_reference.R`
from the repository root. Install `survival` and `jsonlite` with
`install.packages(c("survival", "jsonlite"), repos="https://cloud.r-project.org")`.
The script also accepts an output filename and works from other directories.
The JSON records the installed survival version, input panels, counting-process
intervals, and actual fitted state probabilities. No probabilities are supplied
to the estimator or manually calculated by the generator.

The committed reference was generated with R 4.6.1, survival 3.8.12, and
jsonlite 2.0.0. CI regenerates with the current R release and checks numerical
agreement to an absolute tolerance of 1e-12 rather than requiring version strings
to match. Survival 3.8.12 warns that explicit `type="mstate"` is deprecated in
favor of factor-status inference; the requested explicit form still computes
the multistate fit.

Both examples use the same four-subject A/B/C fixture. Complete follow-up adds
subject 3's B observation at step 2; right censoring omits that observation and
ends their follow-up at 1.5. Events occur at the scheduled observation times.
Intervals are (start, stop], the first event-factor level means censoring,
unchanged states use that level, `istate` supplies the starting state of each
interval, and `id` links intervals for each subject. Baseline is time 0 with
the empirical initial-state distribution. `survfit0()` exposes the fitted
baseline in `pstate`; all JSON occupancy values are extracted from `pstate`.

At step 2, complete follow-up gives (A, B, C) = (0, 0.5, 0.5), while censoring
gives (0, 0.25, 0.75). The latter differs from the observed-only proportions
(0, 1/3, 2/3), which have a different denominator and target.

Polarstate's scheduled-panel estimator uses subjects observed at both ends of
each adjacent-step pair. For these fixtures this agrees with R's continuous
counting-process risk sets: subject 3 is censored strictly before step 2, after
being observed at step 1. This comparison does not establish equivalence for
intermittent missing observations, unknown transition times, delayed entry, or
censoring tied with a transition. In particular R keeps someone censored at an
event time in the risk set for that event, unlike dropping a missing panel pair.
No such tied-censoring assumption is imposed on polarstate.

Run `uv run pytest tests/test_multistate_r_reference.py -v` to compare both
polarstate predictions and chart estimates against R. Set `RSCRIPT` to the
Rscript executable if it is not on PATH; the regeneration test then runs R in
a temporary directory and compares regenerated inputs and numerical results.
Without R, only regeneration is skipped; comparisons always run.

R definitions: [survfit](https://stat.ethz.ch/R-manual/R-devel/library/survival/html/survfit.formula.html),
[Surv](https://stat.ethz.ch/R-manual/R-devel/library/survival/html/Surv.html),
[pstate](https://stat.ethz.ch/R-manual/R-devel/library/survival/html/survfit.object.html).

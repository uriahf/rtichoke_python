# Changelog

<!--next-version-placeholder-->

- Added opt-in canonical browser rendering for static Interventions Avoided using the verified `rtichoke_viz v0.7.0` release while preserving Plotly as the default.
- Fixed Interventions Avoided to apply the per-100 scaling to the full model expression, including the false-negative penalty term.

## v0.1.36 (21/08/2026)

- Fixed several binary and time-dependent curve consistency issues, including cutoff-grid endpoints, binary cutoff equality, time-dependent reference prevalence, gains perfect-reference behavior, custom colors, and plot sizing.
- Added stricter validation for probability/outcome domains and multi-population input alignment, and made binary/time-dependent performance-data ordering and column schemas deterministic.
- Updated time-dependent calibration defaults to use the adjusted-censoring / competing-as-negative heuristic set when omitted, while rejecting ambiguous multiple heuristic sets.
- Moved secondary-Cox and LOWESS calibration smoothing onto `smoothstate`, removing obsolete runtime dependencies including `statsmodels`, `pandas`, `pyarrow`, `typing`, and runtime `marimo`.
- Added Plotly 6 support and raised the supported Python range to 3.12–3.14, with CI now enforcing Ruff lint/formatting and `ty` type checking across the supported interpreters.
- Improved package metadata and project links, and upgraded the documentation toolchain to Great Docs 0.15.0.

## v0.1.33

- Extended `smooth_method="secondary_cox"` in `create_calibration_curve_times` to use 3-knot restricted cubic splines (RCS) on complementary log-log predictions.

## v0.1.0 (27/01/2023)

- First release of `rtichoke`!

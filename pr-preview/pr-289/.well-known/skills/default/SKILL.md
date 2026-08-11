# rtichoke

Use this skill when writing or debugging Python code that evaluates predictive-model performance with `rtichoke`.

## Start here

- Use the generated API reference for signatures and parameter details.
- Use `llms-full.txt` for the complete API plus user-guide content.
- Read **Curve API Compatibility** before assuming that ROC, precision-recall, decision, and calibration functions accept identical inputs.
- Search **Common Errors & Fixes** by literal exception text before source-diving.

## Function families

The main exported families include:

- ROC: `create_roc_curve()`, `create_roc_curve_times()`
- Precision-recall: `create_precision_recall_curve()`, `create_precision_recall_curve_times()`
- Calibration: `create_calibration_curve()`, `create_calibration_curve_times()`
- Decision curve: `create_decision_curve()`, `create_decision_curve_times()`

Similar names do not guarantee identical edge-case behavior.

## Calibration gotchas

1. Matching `probs` / `reals` dictionary keys are paired population-by-population. Different populations may have different sample sizes; lengths only need to match within each population.
2. `create_calibration_curve_times()` currently requires `heuristics_sets`; it does not inherit the default used by ROC/PR/decision `_times` functions.
3. Do not blindly pass `censoring_heuristic="adjusted"` to time-dependent calibration. The current path can skip all horizons and finish with `No data remaining after applying heuristics and time horizons.` The documented working exclusion-based path uses `censoring_heuristic="excluded"` with `competing_heuristic="adjusted_as_negative"`.
4. Prefer floating-point `fixed_time_horizons`, for example `[3.0, 6.0, 9.0]`. Integer horizons can currently leak a Polars `i64` versus `f64` join-key error.

## Debugging rule

When a call that works for ROC/PR/decision fails for calibration, first check whether the failure concerns:

- mismatched population keys or within-population lengths,
- `heuristics_sets`,
- the censoring heuristic,
- or integer time horizons.

## Resources

- Documentation site: https://uriahf.github.io/rtichoke_python/
- Full machine-readable documentation: https://uriahf.github.io/rtichoke_python/llms-full.txt
- Source repository: https://github.com/uriahf/rtichoke_python

# rtichoke

Use this skill when writing or debugging Python code that evaluates predictive-model performance with `rtichoke`.

## Start here

- Use the generated API reference for signatures and parameter details.
- Use `llms-full.txt` for the complete API plus user-guide content.
- Read **Curve API Compatibility** before assuming that all curve families accept identical time-dependent heuristics.
- Search **Common Errors & Fixes** by literal exception text before source-diving.

## Function families

The main exported families include:

- ROC: `create_roc_curve()`, `create_roc_curve_times()`
- Precision-recall: `create_precision_recall_curve()`, `create_precision_recall_curve_times()`
- Gains: `create_gains_curve()`, `create_gains_curve_times()`
- Lift: `create_lift_curve()`, `create_lift_curve_times()`
- Calibration: `create_calibration_curve()`, `create_calibration_curve_times()`
- Decision curve: `create_decision_curve()`, `create_decision_curve_times()`

Similar names do not guarantee identical edge-case behavior.

## Shared input patterns

- Named populations such as Train and Test can be represented by dictionaries. With dictionary-valued outcomes, keys are paired population-by-population and lengths must match within each population; populations themselves may have different sample sizes.
- For time-dependent calls, dictionary-valued `times` follows the same population alignment.
- A censoring heuristic affects estimates only when censored observations are present. A competing-event heuristic affects estimates only when competing events are present. Function-specific validation rules still apply independently of whether a heuristic would change the estimates.
- `fixed_time_horizons` accepts numeric values. Integer horizons such as `[3, 6, 9]` are normalized to floats at the shared time-dependent processing boundary.

## Calibration gotchas

1. `create_calibration_curve_times()` currently requires `heuristics_sets`; it does not inherit the default used by ROC/PR/Gains/Lift/decision `_times` functions.
2. Time-dependent calibration rejects `censoring_heuristic="adjusted"` and `competing_heuristic="adjusted_as_censored"` with an actionable `Unsupported calibration heuristics` error. A supported exclusion-based path uses `censoring_heuristic="excluded"` with `competing_heuristic="adjusted_as_negative"`.

## Debugging rule

When a call that works for another time-dependent curve family fails for calibration, first check whether the failure concerns:

- mismatched population keys or within-population lengths,
- `heuristics_sets`,
- or an unsupported calibration heuristic.

## Resources

- Documentation site: https://uriahf.github.io/rtichoke_python/
- Full machine-readable documentation: https://uriahf.github.io/rtichoke_python/llms-full.txt
- Source repository: https://github.com/uriahf/rtichoke_python

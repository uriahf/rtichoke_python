# Internal population preparation and canonical serialization

`rtichoke._terminal_outcome_distribution` provides internal preparation only;
no public entry point or report uses it. Inputs are a mapping of semantic
population labels to nonempty Polars frames (`id`, `reals`, `times`) and supplied
fixed horizons. IDs must be unique/nonnull within each population. Zero is
evaluated normally when supplied; it is never inserted as a synthetic baseline.
Repeated horizons are deduplicated and sorted.

`_prepare_terminal_outcome_distributions()` returns all eight pairs per population
and horizon, including complete probabilities, estimator, status, retained and
excluded IDs, original/retained sizes, exclusion counts, and exclusion fractions
whose denominator is the original population size. It reuses existing outcome
validation and exclusion extraction. Retention is applied to original outcomes
before recoding. Existing cutoff/stratum producers cannot supply population
denominators and are not used. KM/AJ estimation is delegated to public
`polarstate.prepare_event_table()` and `predict_aj_estimates()`; no estimator
formulas are implemented. Original data and existing statistical helpers are
unchanged. The scalar-risk expected failures from #441 remain applicable.

Population labels denote semantic identity independently of numerical equality.
Evaluation IDs follow the existing outcome adapter's deterministic population
enumeration and remain identical across variants/horizons. No model identity is
inferred for these population-only inputs.

## Published v0.24.1 contract limits

The immutable vendored JS exports `OutcomeDistributionSpecSchema`,
`StateDistributionSchema`, and `assertOutcomeDistributionReferentialIntegrity`.
Tests execute those exports and validate all representable fixture contexts and
a multi-population payload. They do not use rtichoke_viz/main or rebuild the
renderer. Python JSON Schema validation uses the existing dev environment's
jsonschema; Node is needed to execute published integrity checks (CI provides it).

Available variants contain complete `stateDistributions` and `exclusions`.
Probabilities use `estimate`, never integer `count`. `originalSampleSize` and
horizon-specific exclusion counts preserve the original denominator; retained
size is recoverable as original minus both exclusions. The contract has no
explicit retained-size field, so none is invented. The required base distribution
and exclusions are the `adjusted:adjusted_as_negative` variant. No report-level
configuration or controls are added.

The following preparation statuses cannot be faithfully exported for interactive
reports under this contract:

- **Empty retained population:** `estimate: null` passes the schema and integrity
  checks, but there is no per-horizon undefined reason/status. The published
  renderer's state processing starts proportions at zero and leaves them there
  when all estimates/masses/counts are null. Thus schema validity alone does not
  ensure faithful undefined display.
- **Beyond-support plateau:** package values are preserved internally with
  `beyond_support_plateau`, as #441 specifies. The state distribution schema has
  no support-status field to distinguish this from an identified estimate.
- `status: unsupported` exists only for the entire variant, not a particular
  population/horizon, and carries neither its distributions nor exclusions.
  Replacing a whole variant would discard valid horizons and accounting.

`_terminal_outcome_distribution_v2_spec()` therefore raises a diagnostic naming
every unrepresentable evaluation/pair/horizon/status before producing any
payload. It does not omit horizons, substitute zero, encode status in unrelated
fields, invent schema extensions, or mark whole variants unsupported. Internal
preparation remains complete even when export is blocked. Shared contract and
renderer work would be required to remove this limitation; both are outside
this PR.

## Numerical serialization boundary

The package returns `event_free = -1.1102230246251565e-16` for Fixture A's
`excluded:adjusted_as_negative` at horizons 14 and 15. It matches independent R
zero within the reference tolerance, but the published integrity check requires
exact `[0, 1]` bounds. Preparation retains the original output. Serialization
clips only endpoint roundoff within eight floating-point epsilons; larger bound
violations, nonfinite values, or missing estimates for defined rows are errors.
There is no renormalization or recomputation of event probabilities. Independent
reference artifacts and expectations are unchanged.

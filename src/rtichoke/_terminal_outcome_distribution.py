"""Internal terminal-event population preparation and status-aware serialization.

No report entry point uses this module yet. Estimation belongs to polarstate;
population accounting and canonical adaptation are kept separate below.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import product
from typing import Any, Literal

import numpy as np
import polars as pl
from polarstate import predict_aj_estimates, prepare_event_table

from rtichoke.processing.adjustments import _extract_excluded_events
from rtichoke.processing.time_input_validation import _validate_time_outcome_values

_HEURISTIC_PAIRS = tuple(
    product(
        ("excluded", "adjusted"),
        (
            "excluded",
            "adjusted_as_censored",
            "adjusted_as_negative",
            "adjusted_as_composite",
        ),
    )
)
_EstimateStatus = Literal[
    "defined", "undefined_empty_population", "beyond_support_plateau"
]


@dataclass(frozen=True)
class _TerminalOutcomeDistribution:
    evaluation_id: str
    population: str
    horizon: float
    censoring: str
    competing: str
    original_n: int
    retained_ids: tuple[object, ...]
    censored_excluded_ids: tuple[object, ...]
    competing_excluded_ids: tuple[object, ...]
    censored_excluded_count: int
    competing_excluded_count: int
    estimator: str
    status: _EstimateStatus
    probabilities: dict[str, float | None]

    @property
    def retained_n(self) -> int:
        return len(self.retained_ids)

    @property
    def censored_excluded_fraction(self) -> float:
        return self.censored_excluded_count / self.original_n

    @property
    def competing_excluded_fraction(self) -> float:
        return self.competing_excluded_count / self.original_n


def _prepare_terminal_outcome_distributions(
    populations: Mapping[str, pl.DataFrame],
    fixed_time_horizons: Sequence[float],
) -> tuple[_TerminalOutcomeDistribution, ...]:
    """Prepare every pair/population/horizon, preserving subject IDs and status.

    Each population is a nonempty terminal-event frame with unique nonnull `id`,
    outcome `reals` (0/1/2), and finite nonnegative `times`. Only supplied horizons
    are prepared; zero is not injected as a synthetic baseline. Population keys
    denote semantic populations, even when their numerical data are identical.
    """
    if not populations:
        raise ValueError("At least one population is required.")
    horizons = np.asarray(fixed_time_horizons, dtype=float)
    if (
        horizons.ndim != 1
        or not horizons.size
        or not np.all(np.isfinite(horizons))
        or np.any(horizons < 0)
    ):
        raise ValueError(
            "Fixed horizons must be a nonempty sequence of finite nonnegative numbers."
        )
    ordered_horizons = sorted(set(horizons.tolist()))
    result = []
    for index, (population, source) in enumerate(populations.items(), start=1):
        if not isinstance(population, str) or not population:
            raise ValueError("Population keys must be nonempty strings.")
        if not {"id", "reals", "times"} <= set(source.columns) or source.is_empty():
            raise ValueError(
                "Each population must be nonempty and contain id, reals, and times."
            )
        if source["id"].null_count() or source["id"].n_unique() != source.height:
            raise ValueError(
                "Subject IDs must be nonnull and unique within each population."
            )
        _validate_time_outcome_values(source["reals"].to_numpy())
        times = source["times"].to_numpy()
        if (
            not np.issubdtype(times.dtype, np.number)
            or not np.all(np.isfinite(times))
            or np.any(times < 0)
        ):
            raise ValueError("Follow-up times must be finite nonnegative numbers.")
        # Float horizons and event-table join keys must have matching types.
        original = source.select(
            "id", pl.col("reals").cast(pl.Int64), pl.col("times").cast(pl.Float64)
        )
        event_table = prepare_event_table(original.select("reals", "times"))
        for censoring, competing in _HEURISTIC_PAIRS:
            exclusions = _extract_excluded_events(
                event_table, ordered_horizons, censoring, competing
            )
            for counts in exclusions.iter_rows(named=True):
                h = float(counts["fixed_time_horizon"])
                censor_mask = (
                    (pl.col("reals") == 0)
                    & (pl.col("times") < h)
                    & pl.lit(censoring == "excluded")
                )
                competing_mask = (
                    (pl.col("reals") == 2)
                    & (pl.col("times") <= h)
                    & pl.lit(competing == "excluded")
                )
                # Both masks refer to original outcomes, before any recoding.
                retained = original.filter(~(censor_mask | competing_mask))
                censor_ids = tuple(original.filter(censor_mask)["id"].to_list())
                competing_ids = tuple(original.filter(competing_mask)["id"].to_list())
                states = (
                    ("primary", "competing", "event_free")
                    if competing == "adjusted_as_negative"
                    else (
                        "composite"
                        if competing == "adjusted_as_composite"
                        else "primary",
                        "event_free",
                    )
                )
                estimator = (
                    "aalen_johansen"
                    if competing == "adjusted_as_negative"
                    else "kaplan_meier"
                )
                status: _EstimateStatus = "undefined_empty_population"
                probabilities: dict[str, float | None] = dict.fromkeys(states)
                if retained.height:
                    adjusted = retained.select("reals", "times")
                    if competing in ("adjusted_as_censored", "adjusted_as_composite"):
                        adjusted = adjusted.with_columns(
                            pl.when(pl.col("reals") == 2)
                            .then(0 if competing == "adjusted_as_censored" else 1)
                            .otherwise(pl.col("reals"))
                            .alias("reals")
                        )
                    predicted = predict_aj_estimates(
                        prepare_event_table(adjusted),
                        pl.Series([h]),
                        full_event_table=False,
                    ).row(0, named=True)
                    # Structural cause mapping only; no KM/AJ formulas or estimated counts.
                    columns = (
                        (1, 2, 0) if competing == "adjusted_as_negative" else (1, 0)
                    )
                    probabilities = {
                        state: float(predicted[f"state_occupancy_probability_{cause}"])
                        for state, cause in zip(states, columns, strict=True)
                    }
                    event_free = probabilities["event_free"]
                    status = (
                        "beyond_support_plateau"
                        if h > float(np.max(retained["times"].to_numpy()))
                        and event_free is not None
                        and event_free > 0
                        else "defined"
                    )
                result.append(
                    _TerminalOutcomeDistribution(
                        evaluation_id=f"evaluation-{index}",
                        population=population,
                        horizon=h,
                        censoring=censoring,
                        competing=competing,
                        original_n=original.height,
                        retained_ids=tuple(retained["id"].to_list()),
                        censored_excluded_ids=censor_ids,
                        competing_excluded_ids=competing_ids,
                        censored_excluded_count=int(counts["real_censored_est"]),
                        competing_excluded_count=int(counts["real_competing_est"]),
                        estimator=estimator,
                        status=status,
                        probabilities=probabilities,
                    )
                )
    return tuple(result)


def _canonical_probability(value: float | None) -> float:
    """Bound floating-point endpoint roundoff without changing prepared estimates."""
    tolerance = 8 * np.finfo(float).eps
    if (
        value is None
        or not np.isfinite(value)
        or value < -tolerance
        or value > 1 + tolerance
    ):
        raise ValueError(
            "A defined canonical estimate must be a finite probability in [0, 1]."
        )
    return min(1.0, max(0.0, value))


def _terminal_outcome_distribution_v2_spec(
    prepared: Sequence[_TerminalOutcomeDistribution],
) -> dict[str, Any]:
    """Serialize every supplied context with its canonical estimate status."""
    if not prepared:
        raise ValueError("Prepared distributions must not be empty.")
    row_keys = {
        (r.evaluation_id, r.horizon, r.censoring, r.competing) for r in prepared
    }
    contexts = {(r.evaluation_id, r.horizon) for r in prepared}
    expected_keys = {
        (evaluation, h, c, k) for evaluation, h in contexts for c, k in _HEURISTIC_PAIRS
    }
    if len(row_keys) != len(prepared) or row_keys != expected_keys:
        raise ValueError(
            "Serialization requires exactly eight pairs per population and horizon."
        )
    evaluations = list(
        {
            r.evaluation_id: {"id": r.evaluation_id, "population": r.population}
            for r in prepared
        }.values()
    )
    variants = {}
    for censoring, competing in _HEURISTIC_PAIRS:
        rows = [
            r for r in prepared if (r.censoring, r.competing) == (censoring, competing)
        ]
        distributions = [
            {
                "evaluationId": r.evaluation_id,
                "horizon": r.horizon,
                "estimator": r.estimator,
                "estimateOrigin": "fixed_time_horizon",
                "estimateStatus": r.status,
                "censoringHeuristic": r.censoring,
                "competingHeuristic": r.competing,
                "states": [
                    {
                        "stateId": state,
                        "label": {
                            "primary": "Primary event",
                            "competing": "Competing event",
                            "composite": "Composite event",
                            "event_free": "Event-free",
                        }[state],
                        "estimate": (
                            None
                            if estimate is None
                            and r.status == "undefined_empty_population"
                            else _canonical_probability(estimate)
                        ),
                    }
                    for state, estimate in r.probabilities.items()
                ],
            }
            for r in rows
        ]
        exclusions = [
            {
                "evaluationId": evaluation["id"],
                "originalSampleSize": next(
                    r.original_n for r in rows if r.evaluation_id == evaluation["id"]
                ),
                "excludedCounts": [
                    {
                        "horizon": r.horizon,
                        "censoredExcludedCount": r.censored_excluded_count,
                        "competingExcludedCount": r.competing_excluded_count,
                    }
                    for r in rows
                    if r.evaluation_id == evaluation["id"]
                ],
            }
            for evaluation in evaluations
        ]
        variants[f"{censoring}:{competing}"] = {
            "status": "available",
            "stateDistributions": distributions,
            "exclusions": exclusions,
        }
    # The required legacy/base payload is the same explicit pair as this entry.
    base = variants["adjusted:adjusted_as_negative"]
    return {
        "schemaVersion": "2.0",
        "type": "outcome_distribution",
        "title": "Outcome Distribution",
        "evaluations": evaluations,
        "stateDistributions": base["stateDistributions"],
        "exclusions": base["exclusions"],
        "variants": variants,
    }

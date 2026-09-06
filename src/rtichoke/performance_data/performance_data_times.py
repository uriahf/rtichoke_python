"""
A module for Performance Data with Time Dimension
"""

from typing import Dict, Union
import polars as pl
from collections.abc import Sequence
from rtichoke.processing.adjustments import create_adjusted_data
from rtichoke.processing.combinations import (
    create_aj_data_combinations,
    create_breaks_values,
)
from rtichoke.processing.time_input_validation import _validate_time_input_alignment
from rtichoke.processing.transforms import (
    _calculate_cumulative_aj_data,
    _create_list_data_to_adjust,
    _turn_cumulative_aj_to_performance_data,
    cast_and_join_adjusted_data,
)

import numpy as np
from polarstate import predict_aj_estimates, prepare_event_table


_PERFORMANCE_DATA_TIMES_COLUMNS = [
    "reference_group",
    "fixed_time_horizon",
    "censoring_heuristic",
    "competing_heuristic",
    "stratified_by",
    "chosen_cutoff",
    "excluded",
    "true_positives",
    "true_negatives",
    "false_positives",
    "false_negatives",
    "predicted_positives",
    "predicted_negatives",
    "real_positives",
    "real_negatives",
    "n",
    "sensitivity",
    "specificity",
    "ppv",
    "npv",
    "false_positive_rate",
    "lift",
    "net_benefit",
    "net_benefit_interventions_avoided",
    "ppcr",
]


def prepare_performance_data_times(
    probs: Dict[str, np.ndarray],
    reals: Union[np.ndarray, Dict[str, np.ndarray]],
    times: Union[np.ndarray, Dict[str, np.ndarray]],
    fixed_time_horizons: list[float],
    heuristics_sets: list[Dict] = [
        {
            "censoring_heuristic": "adjusted",
            "competing_heuristic": "adjusted_as_negative",
        }
    ],
    stratified_by: Sequence[str] = ("probability_threshold",),
    by: float = 0.01,
) -> pl.DataFrame:
    """Prepare performance data for models with time-to-event outcomes.

    This function calculates a comprehensive set of performance metrics for
    models predicting time-to-event outcomes. It handles censored data and
    competing events by applying specified heuristics at different time
    horizons. The function first bins the data using
    `prepare_binned_classification_data_times` and then computes cumulative,
    Aalen-Johansen-based performance metrics.

    The resulting dataframe is the primary input for time-dependent plotting
    functions.

    Parameters
    ----------
    probs : Dict[str, np.ndarray]
        A dictionary mapping model or dataset names (str) to their predicted
        probabilities of an event occurring by a given time.
    reals : Union[np.ndarray, Dict[str, np.ndarray]]
        The true event statuses. Can be a single array or a dictionary.
        Labels should be integers indicating the outcome (e.g., 0=censored,
        1=event of interest, 2=competing event).
    times : Union[np.ndarray, Dict[str, np.ndarray]]
        The event or censoring times corresponding to the `reals`. Can be a
        single array or a dictionary.
    fixed_time_horizons : list[float]
        A list of numeric time points at which to evaluate the model's
        performance. Integer inputs are accepted and normalized to floats.
    heuristics_sets : list[Dict], optional
        A list of dictionaries, each specifying how to handle censored data
        and competing events. The default is
        ``[{"censoring_heuristic": "adjusted",
        "competing_heuristic": "adjusted_as_negative"}]``.
    stratified_by : Sequence[str], optional
        Variables by which to stratify the analysis. Defaults to
        ``("probability_threshold",)``.
    by : float, optional
        The step size for probability thresholds. Defaults to ``0.01``.

    Returns
    -------
    pl.DataFrame
        A Polars DataFrame with performance metrics computed across probability
        thresholds and time horizons. It includes columns for cutoffs, time
        points, heuristics, and performance measures.
    """
    final_adjusted_data = prepare_binned_classification_data_times(
        probs=probs,
        reals=reals,
        times=times,
        fixed_time_horizons=fixed_time_horizons,
        heuristics_sets=heuristics_sets,
        stratified_by=stratified_by,
        by=by,
        risk_set_scope=["pooled_by_cutoff"],
    )

    cumulative_aj_data = _calculate_cumulative_aj_data(final_adjusted_data)
    performance_data = _turn_cumulative_aj_to_performance_data(cumulative_aj_data)
    performance_data = _recalculate_interventions_avoided_times(
        performance_data,
        probs=probs,
        reals=reals,
        times=times,
        fixed_time_horizons=fixed_time_horizons,
        heuristics_sets=heuristics_sets,
    )

    group_order = {group: index for index, group in enumerate(probs)}
    horizon_order = {
        float(horizon): index for index, horizon in enumerate(fixed_time_horizons)
    }
    heuristic_order = {
        f"{heuristics['censoring_heuristic']}\x1f{heuristics['competing_heuristic']}": index
        for index, heuristics in enumerate(heuristics_sets)
    }

    return (
        performance_data.with_columns(
            pl.col("reference_group")
            .replace_strict(group_order, default=len(group_order))
            .alias("_reference_group_order"),
            pl.col("fixed_time_horizon")
            .replace_strict(horizon_order, default=len(horizon_order))
            .alias("_fixed_time_horizon_order"),
            pl.concat_str(
                ["censoring_heuristic", "competing_heuristic"], separator="\x1f"
            )
            .replace_strict(heuristic_order, default=len(heuristic_order))
            .alias("_heuristic_order"),
        )
        .sort(
            [
                "_fixed_time_horizon_order",
                "_heuristic_order",
                "stratified_by",
                "chosen_cutoff",
                "_reference_group_order",
            ]
        )
        .drop(
            "_reference_group_order",
            "_fixed_time_horizon_order",
            "_heuristic_order",
        )
        .select(_PERFORMANCE_DATA_TIMES_COLUMNS)
    )


def _compute_population_event_risk_times(
    reals: np.ndarray,
    times: np.ndarray,
    horizon: float,
    censoring_heuristic: str,
    competing_heuristic: str,
) -> float:
    """Compute the genuine pooled full-population KM/AJ event risk estimate."""
    df = pl.DataFrame(
        {
            "reals": np.asarray(reals),
            "times": np.asarray(times, dtype=float),
            "fixed_time_horizon": float(horizon),
        }
    )

    if censoring_heuristic == "excluded":
        df = df.filter(
            (pl.col("times") > pl.col("fixed_time_horizon")) | (pl.col("reals") > 0)
        )

    if competing_heuristic == "excluded":
        df = df.filter(
            (pl.col("times") > pl.col("fixed_time_horizon")) | (pl.col("reals") != 2)
        )
    elif competing_heuristic == "adjusted_as_censored":
        df = df.with_columns(
            pl.when(pl.col("reals") == 2)
            .then(0)
            .otherwise(pl.col("reals"))
            .alias("reals")
        )
    elif competing_heuristic == "adjusted_as_composite":
        df = df.with_columns(
            pl.when(pl.col("reals") == 2)
            .then(1)
            .otherwise(pl.col("reals"))
            .alias("reals")
        )

    event_table = prepare_event_table(df)
    estimate = predict_aj_estimates(
        event_table, pl.Series([float(horizon)]), full_event_table=False
    )
    return float(estimate["state_occupancy_probability_1"][0])


def _recalculate_interventions_avoided_times(
    performance_data: pl.DataFrame,
    probs: Dict[str, np.ndarray],
    reals: Union[np.ndarray, Dict[str, np.ndarray]],
    times: Union[np.ndarray, Dict[str, np.ndarray]],
    fixed_time_horizons: list[float],
    heuristics_sets: list[Dict],
) -> pl.DataFrame:
    """Recalculate time-dependent interventions avoided using model and treat-all net benefit.

    IA = 100 * (NB_model - NB_all) / [threshold / (1 - threshold)]

    Population event risk is computed from the full population dataset for each group,
    horizon, censoring heuristic, and competing heuristic, independent of prediction values.
    Interventions avoided is calculated for probability_threshold rows where 0 < chosen_cutoff < 1.
    For chosen_cutoff == 0 or 1, and for PPCR rows, interventions avoided is set to null.
    """
    rows = []
    for group in probs:
        reals_group = reals[group] if isinstance(reals, dict) else reals
        times_group = times[group] if isinstance(times, dict) else times
        for horizon in fixed_time_horizons:
            for heuristics in heuristics_sets:
                censoring = heuristics["censoring_heuristic"]
                competing = heuristics["competing_heuristic"]
                risk = _compute_population_event_risk_times(
                    reals_group, times_group, horizon, censoring, competing
                )
                rows.append(
                    {
                        "reference_group": group,
                        "fixed_time_horizon": float(horizon),
                        "censoring_heuristic": censoring,
                        "competing_heuristic": competing,
                        "_event_risk": risk,
                    }
                )

    event_risk_df = pl.DataFrame(rows)
    for col in ["reference_group", "censoring_heuristic", "competing_heuristic"]:
        if col in performance_data.columns and col in event_risk_df.columns:
            event_risk_df = event_risk_df.with_columns(
                pl.col(col).cast(performance_data.schema[col])
            )

    performance_data = performance_data.join(
        event_risk_df,
        on=[
            "reference_group",
            "fixed_time_horizon",
            "censoring_heuristic",
            "competing_heuristic",
        ],
        how="left",
    )

    threshold_odds = pl.col("chosen_cutoff") / (1 - pl.col("chosen_cutoff"))
    net_benefit_all = (
        pl.col("_event_risk") - (1 - pl.col("_event_risk")) * threshold_odds
    )

    ia_expr = (
        pl.when(
            (pl.col("stratified_by") == "probability_threshold")
            & (pl.col("chosen_cutoff") > 0)
            & (pl.col("chosen_cutoff") < 1)
        )
        .then(
            100
            * (pl.col("net_benefit") - net_benefit_all)
            * (1 - pl.col("chosen_cutoff"))
            / pl.col("chosen_cutoff")
        )
        .otherwise(None)
    )

    return performance_data.with_columns(
        ia_expr.alias("net_benefit_interventions_avoided")
    ).drop("_event_risk")


def prepare_binned_classification_data_times(
    probs: Dict[str, np.ndarray],
    reals: Union[np.ndarray, Dict[str, np.ndarray]],
    times: Union[np.ndarray, Dict[str, np.ndarray]],
    fixed_time_horizons: list[float],
    heuristics_sets: list[Dict] = [
        {
            "censoring_heuristic": "adjusted",
            "competing_heuristic": "adjusted_as_negative",
        }
    ],
    stratified_by: Sequence[str] = ("probability_threshold",),
    by: float = 0.01,
    risk_set_scope: Sequence[str] = ["pooled_by_cutoff", "within_stratum"],
) -> pl.DataFrame:
    """
    Prepare binned, time-dependent classification data.

    This function constructs the foundational binned data needed for
    time-to-event performance analysis. It bins predictions by probability
    thresholds, applies censoring and competing event heuristics, and stratifies
    the data across specified time horizons. The output is a detailed breakdown
    of outcomes within each bin, which can be used for calibration or passed to
    `prepare_performance_data_times` for full performance metric calculation.

    Parameters
    ----------
    probs : Dict[str, np.ndarray]
        A dictionary mapping model or dataset names (str) to their predicted
        probabilities.
    reals : Union[np.ndarray, Dict[str, np.ndarray]]
        The true event statuses (e.g., 0=censored, 1=event, 2=competing event).
    times : Union[np.ndarray, Dict[str, np.ndarray]]
        The event or censoring times.
    fixed_time_horizons : list[float]
        A list of numeric time points for performance evaluation. Integer
        inputs are accepted and normalized to floats.
    heuristics_sets : list[Dict], optional
        Specifies how to handle censored data and competing events.
    stratified_by : Sequence[str], optional
        Variables for stratification. Defaults to ``("probability_threshold",)``.
    by : float, optional
        The step size for probability thresholds. Defaults to ``0.01``.
    risk_set_scope : Sequence[str], optional
        Defines the scope for risk set calculations. Defaults to
        ``["pooled_by_cutoff", "within_stratum"]``.

    Returns
    -------
    pl.DataFrame
        A Polars DataFrame with binned, time-dependent data. Each row
        represents a unique combination of dataset, bin, time horizon,
        heuristic, and other strata.
    """
    _validate_time_input_alignment(probs=probs, reals=reals, times=times)
    fixed_time_horizons = [float(horizon) for horizon in fixed_time_horizons]

    breaks = create_breaks_values(None, "probability_threshold", by)

    aj_data_combinations = create_aj_data_combinations(
        list(probs.keys()),
        heuristics_sets=heuristics_sets,
        fixed_time_horizons=fixed_time_horizons,
        stratified_by=stratified_by,
        by=by,
        breaks=breaks,
        risk_set_scope=risk_set_scope,
    )

    list_data_to_adjust = _create_list_data_to_adjust(
        aj_data_combinations,
        probs,
        reals,
        times,
        stratified_by=stratified_by,
        by=by,
    )

    adjusted_data = create_adjusted_data(
        list_data_to_adjust,
        heuristics_sets=heuristics_sets,
        fixed_time_horizons=fixed_time_horizons,
        breaks=breaks,
        stratified_by=stratified_by,
        risk_set_scope=risk_set_scope,
    )

    final_adjusted_data = cast_and_join_adjusted_data(
        aj_data_combinations,
        adjusted_data,
    ).with_columns(pl.col("reals_estimate").fill_null(0.0))

    return final_adjusted_data

"""Private producer for prediction distribution data reproducing static binary semantics."""

from typing import Dict, Sequence, TypedDict, Union
import numpy as np
import polars as pl

from rtichoke.performance_data.performance_data import (
    _validate_and_align_binary_inputs,
    prepare_performance_data,
)
from rtichoke.processing.evaluation_semantics import (
    _EvaluationMetadata,
    _build_evaluation_metadata,
)
from rtichoke.processing.transforms import _compute_probability_quantile_bin_indices


class _PredictionDistributionData(TypedDict):
    bins: pl.DataFrame
    operating_points: pl.DataFrame
    rank_bins: pl.DataFrame


def _aggregate_rank_bins_for_evaluation(
    probabilities: np.ndarray,
    outcomes: np.ndarray,
    by: float,
    evaluation_metadata: _EvaluationMetadata,
) -> pl.DataFrame:
    """Aggregate observed positive and negative mass into probability-quantile rank bins."""
    by = float(by)
    q = int(round(1 / by))
    grid_rows = []
    for i in range(q):
        grid_rows.append(
            {
                "stratum_id": i,
                "evaluation": evaluation_metadata.evaluation,
                "model": evaluation_metadata.model,
                "population": evaluation_metadata.population,
                "rank_lower": float(round(i * by, 10)),
                "rank_upper": float(round((i + 1) * by, 10)),
            }
        )

    grid_schema = {
        "stratum_id": pl.Int64,
        "evaluation": pl.String,
        "model": pl.String,
        "population": pl.String,
        "rank_lower": pl.Float64,
        "rank_upper": pl.Float64,
    }

    complete_grid = pl.DataFrame(grid_rows, schema=grid_schema)

    if len(probabilities) == 0:
        return complete_grid.with_columns(
            pl.lit(0, dtype=pl.Int64).alias("n_positive"),
            pl.lit(0, dtype=pl.Int64).alias("n_negative"),
        ).drop("stratum_id")

    bin_indices, _ = _compute_probability_quantile_bin_indices(probabilities, by)

    obs_df = pl.DataFrame(
        {
            "stratum_id": bin_indices,
            "is_pos": (outcomes == 1).astype(int),
            "is_neg": (outcomes == 0).astype(int),
        }
    )

    counts_df = obs_df.group_by("stratum_id").agg(
        pl.col("is_pos").sum().cast(pl.Int64).alias("n_positive"),
        pl.col("is_neg").sum().cast(pl.Int64).alias("n_negative"),
    )

    aggregated_rank_bins = (
        complete_grid.join(counts_df, on="stratum_id", how="left")
        .with_columns(
            pl.col("n_positive").fill_null(0),
            pl.col("n_negative").fill_null(0),
        )
        .drop("stratum_id")
    )

    return aggregated_rank_bins


def _aggregate_bins_for_evaluation(
    probabilities: np.ndarray,
    outcomes: np.ndarray,
    interval_boundaries: np.ndarray,
    evaluation_metadata: _EvaluationMetadata,
) -> pl.DataFrame:
    """Aggregate observation counts into complete interval grid for one evaluation."""
    # Build complete interval grid
    # Interval 0 is [0.0, 0.0]
    # Intervals 1..k are (interval_boundaries[i-1], interval_boundaries[i]]
    grid_rows = [
        {
            "interval_id": 0,
            "evaluation": evaluation_metadata.evaluation,
            "model": evaluation_metadata.model,
            "population": evaluation_metadata.population,
            "lower": 0.0,
            "upper": 0.0,
            "include_lower": True,
            "include_upper": True,
        }
    ]

    for i in range(1, len(interval_boundaries)):
        grid_rows.append(
            {
                "interval_id": i,
                "evaluation": evaluation_metadata.evaluation,
                "model": evaluation_metadata.model,
                "population": evaluation_metadata.population,
                "lower": float(interval_boundaries[i - 1]),
                "upper": float(interval_boundaries[i]),
                "include_lower": False,
                "include_upper": True,
            }
        )

    grid_schema = {
        "interval_id": pl.Int64,
        "evaluation": pl.String,
        "model": pl.String,
        "population": pl.String,
        "lower": pl.Float64,
        "upper": pl.Float64,
        "include_lower": pl.Boolean,
        "include_upper": pl.Boolean,
    }

    complete_grid = pl.DataFrame(grid_rows, schema=grid_schema)

    if len(probabilities) == 0:
        return complete_grid.with_columns(
            pl.lit(0, dtype=pl.UInt32).alias("n_positive"),
            pl.lit(0, dtype=pl.UInt32).alias("n_negative"),
        ).drop("interval_id")

    # Determine interval_id for each observation:
    # prob == 0 -> interval_id = 0
    # prob > 0 -> interval_id via np.digitize(prob, interval_boundaries, right=True)
    zero_score_mask = probabilities == 0.0
    interval_ids = np.zeros(len(probabilities), dtype=int)

    if np.any(~zero_score_mask):
        interval_ids[~zero_score_mask] = np.digitize(
            probabilities[~zero_score_mask], interval_boundaries, right=True
        )

    obs_df = pl.DataFrame(
        {
            "interval_id": interval_ids,
            "is_pos": (outcomes == 1).astype(int),
            "is_neg": (outcomes == 0).astype(int),
        }
    )

    counts_df = obs_df.group_by("interval_id").agg(
        pl.col("is_pos").sum().cast(pl.UInt32).alias("n_positive"),
        pl.col("is_neg").sum().cast(pl.UInt32).alias("n_negative"),
    )

    aggregated_bins = (
        complete_grid.join(counts_df, on="interval_id", how="left")
        .with_columns(
            pl.col("n_positive").fill_null(0),
            pl.col("n_negative").fill_null(0),
        )
        .drop("interval_id")
    )

    return aggregated_bins


def _prepare_probs_distribution_data(
    probs: Dict[str, np.ndarray],
    reals: Union[np.ndarray, Dict[str, np.ndarray]],
    stratified_by: Sequence[str] = ("probability_threshold",),
    by: float = 0.01,
) -> _PredictionDistributionData:
    """Prepare internal prediction distribution bins and operating points.

    Parameters
    ----------
    probs : Dict[str, np.ndarray]
        Dictionary mapping model or evaluation names to predicted probabilities.
    reals : Union[np.ndarray, Dict[str, np.ndarray]]
        True binary labels (0 or 1), as array or dictionary matching probs keys.
    stratified_by : Sequence[str], optional
        Sequence containing exactly one stratification key, either
        ``("probability_threshold",)`` or ``("ppcr",)``.
    by : float, optional
        Step size for grid generation. Defaults to ``0.01``.

    Returns
    -------
    _PredictionDistributionData
        TypedDict containing ``bins``, ``operating_points``, and ``rank_bins``
        Polars DataFrames. In ``operating_points``, ``value`` is the requested grid
        point (requested PPCR or probability threshold), ``cutoff`` is the effective
        predicted-probability boundary (from ``probability_threshold``), and
        ``realized_ppcr`` is the empirical proportion classified positive.
    """
    if not isinstance(stratified_by, (list, tuple)) or len(stratified_by) != 1:
        raise ValueError(
            "`stratified_by` must be a sequence containing exactly one element: "
            "'probability_threshold' or 'ppcr'."
        )

    stratification_type = stratified_by[0]
    if stratification_type not in ("probability_threshold", "ppcr"):
        raise ValueError(
            f"Unsupported stratification key {stratification_type!r}. "
            "Must be 'probability_threshold' or 'ppcr'."
        )

    aligned_reals = _validate_and_align_binary_inputs(probs=probs, reals=reals)

    # Derive evaluation metadata from original reals to preserve single keyed-population semantics
    dummy_times = np.array([])
    evaluation_metadata_by_group = _build_evaluation_metadata(probs, reals, dummy_times)

    evaluation_ids = [
        metadata.evaluation for metadata in evaluation_metadata_by_group.values()
    ]
    if len(evaluation_ids) != len(set(evaluation_ids)):
        raise ValueError("Duplicate evaluation identifiers detected.")

    evaluation_keys = list(evaluation_metadata_by_group.keys())

    # Call authoritative production performance data
    performance_data = prepare_performance_data(
        probs=probs,
        reals=aligned_reals,
        stratified_by=stratified_by,
        by=by,
    )

    operating_point_schema = {
        "evaluation": pl.String,
        "model": pl.String,
        "population": pl.String,
        "type": pl.String,
        "value": pl.Float64,
        "cutoff": pl.Float64,
        "realized_ppcr": pl.Float64,
    }

    eval_bins_frames = []
    operating_point_rows = []

    for evaluation_key in evaluation_keys:
        evaluation_metadata = evaluation_metadata_by_group[evaluation_key]

        evaluation_performance_data = performance_data.filter(
            pl.col("reference_group") == evaluation_key
        )

        probabilities = np.asarray(probs[evaluation_key], dtype=float)
        if isinstance(aligned_reals, dict):
            outcomes = np.asarray(aligned_reals[evaluation_key], dtype=int)
        else:
            outcomes = np.asarray(aligned_reals, dtype=int)

        # Build operating points rows
        for row in evaluation_performance_data.iter_rows(named=True):
            requested_value = float(
                row["ppcr"] if stratification_type == "ppcr" else row["chosen_cutoff"]
            )
            effective_cutoff = float(row["probability_threshold"])
            n_observations = int(row["n"])
            predicted_positives = int(row["predicted_positives"])
            realized_ppcr = (
                float(predicted_positives / n_observations)
                if n_observations > 0
                else 0.0
            )

            operating_point_rows.append(
                {
                    "evaluation": evaluation_metadata.evaluation,
                    "model": evaluation_metadata.model,
                    "population": evaluation_metadata.population,
                    "type": stratification_type,
                    "value": requested_value,
                    "cutoff": effective_cutoff,
                    "realized_ppcr": realized_ppcr,
                }
            )

        # Build interval boundaries from effective cutoffs
        cutoffs = (
            evaluation_performance_data["probability_threshold"]
            .to_numpy()
            .astype(float)
        )
        interval_boundaries = np.unique(np.concatenate(([0.0, 1.0], cutoffs)))
        interval_boundaries.sort()

        eval_bins = _aggregate_bins_for_evaluation(
            probabilities=probabilities,
            outcomes=outcomes,
            interval_boundaries=interval_boundaries,
            evaluation_metadata=evaluation_metadata,
        )
        eval_bins_frames.append(eval_bins)

    eval_rank_bins_frames = []
    for evaluation_key in evaluation_keys:
        evaluation_metadata = evaluation_metadata_by_group[evaluation_key]
        probabilities = np.asarray(probs[evaluation_key], dtype=float)
        if isinstance(aligned_reals, dict):
            outcomes = np.asarray(aligned_reals[evaluation_key], dtype=int)
        else:
            outcomes = np.asarray(aligned_reals, dtype=int)

        eval_rank_bins = _aggregate_rank_bins_for_evaluation(
            probabilities=probabilities,
            outcomes=outcomes,
            by=by,
            evaluation_metadata=evaluation_metadata,
        )
        eval_rank_bins_frames.append(eval_rank_bins)

    bins = pl.concat(eval_bins_frames, how="vertical")
    operating_points = pl.DataFrame(operating_point_rows, schema=operating_point_schema)
    rank_bins = pl.concat(eval_rank_bins_frames, how="vertical")

    return _PredictionDistributionData(
        bins=bins,
        operating_points=operating_points,
        rank_bins=rank_bins,
    )

"""Private producer for prediction distribution data reproducing static binary semantics."""

from typing import Dict, Sequence, TypedDict, Union
import numpy as np
import polars as pl

from rtichoke.performance_data.performance_data import (
    _validate_and_align_binary_inputs,
    prepare_performance_data,
)
from rtichoke.performance_data.performance_data_times import (
    prepare_binned_classification_data_times,
    prepare_performance_data_times,
)
from rtichoke.processing.evaluation_semantics import (
    _EvaluationMetadata,
    _build_evaluation_metadata,
)
from rtichoke.processing.time_input_validation import _validate_time_input_alignment
from rtichoke.processing.transforms import _compute_probability_quantile_bin_indices


class _PredictionDistributionData(TypedDict):
    bins: pl.DataFrame
    operating_points: pl.DataFrame
    rank_bins: pl.DataFrame


class _PredictionDistributionTimesData(TypedDict):
    bins: pl.DataFrame
    operating_points: pl.DataFrame
    rank_bins: pl.DataFrame
    cutoff_region_aj: pl.DataFrame


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


def _aggregate_bins_for_evaluation_times(
    probabilities: np.ndarray,
    outcomes: np.ndarray,
    interval_boundaries: np.ndarray,
    evaluation_metadata: _EvaluationMetadata,
) -> pl.DataFrame:
    """Aggregate raw time-to-event observation counts into interval grid for one evaluation."""
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
            pl.lit(0, dtype=pl.Int64).alias("n_observations"),
            pl.lit(0, dtype=pl.Int64).alias("n_real_positive"),
            pl.lit(0, dtype=pl.Int64).alias("n_real_negative"),
            pl.lit(0, dtype=pl.Int64).alias("n_real_competing"),
        ).drop("interval_id")

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
            "is_comp": (outcomes == 2).astype(int),
        }
    )

    counts_df = obs_df.group_by("interval_id").agg(
        pl.len().cast(pl.Int64).alias("n_observations"),
        pl.col("is_pos").sum().cast(pl.Int64).alias("n_real_positive"),
        pl.col("is_neg").sum().cast(pl.Int64).alias("n_real_negative"),
        pl.col("is_comp").sum().cast(pl.Int64).alias("n_real_competing"),
    )

    aggregated_bins = (
        complete_grid.join(counts_df, on="interval_id", how="left")
        .with_columns(
            pl.col("n_observations").fill_null(0),
            pl.col("n_real_positive").fill_null(0),
            pl.col("n_real_negative").fill_null(0),
            pl.col("n_real_competing").fill_null(0),
        )
        .drop("interval_id")
    )

    return aggregated_bins


def _aggregate_rank_bins_for_evaluation_times(
    probabilities: np.ndarray,
    outcomes: np.ndarray,
    by: float,
    evaluation_metadata: _EvaluationMetadata,
) -> pl.DataFrame:
    """Aggregate raw time-to-event observation counts into probability-quantile rank bins."""
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
            pl.lit(0, dtype=pl.Int64).alias("n_observations"),
            pl.lit(0, dtype=pl.Int64).alias("n_real_positive"),
            pl.lit(0, dtype=pl.Int64).alias("n_real_negative"),
            pl.lit(0, dtype=pl.Int64).alias("n_real_competing"),
        ).drop("stratum_id")

    bin_indices, _ = _compute_probability_quantile_bin_indices(probabilities, by)

    obs_df = pl.DataFrame(
        {
            "stratum_id": bin_indices,
            "is_pos": (outcomes == 1).astype(int),
            "is_neg": (outcomes == 0).astype(int),
            "is_comp": (outcomes == 2).astype(int),
        }
    )

    counts_df = obs_df.group_by("stratum_id").agg(
        pl.len().cast(pl.Int64).alias("n_observations"),
        pl.col("is_pos").sum().cast(pl.Int64).alias("n_real_positive"),
        pl.col("is_neg").sum().cast(pl.Int64).alias("n_real_negative"),
        pl.col("is_comp").sum().cast(pl.Int64).alias("n_real_competing"),
    )

    aggregated_rank_bins = (
        complete_grid.join(counts_df, on="stratum_id", how="left")
        .with_columns(
            pl.col("n_observations").fill_null(0),
            pl.col("n_real_positive").fill_null(0),
            pl.col("n_real_negative").fill_null(0),
            pl.col("n_real_competing").fill_null(0),
        )
        .drop("stratum_id")
    )

    return aggregated_rank_bins


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


def _prepare_probs_distribution_data_times(
    probs: Dict[str, np.ndarray],
    reals: Union[np.ndarray, Dict[str, np.ndarray]],
    times: Union[np.ndarray, Dict[str, np.ndarray]],
    fixed_time_horizons: Sequence[float],
    heuristics_sets: Union[Sequence[Dict[str, str]], None] = None,
    stratified_by: Sequence[str] = ("probability_threshold",),
    by: float = 0.01,
) -> _PredictionDistributionTimesData:
    """Prepare internal time-dependent prediction distribution data.

    Parameters
    ----------
    probs : Dict[str, np.ndarray]
        Dictionary mapping model or evaluation names to predicted probabilities.
    reals : Union[np.ndarray, Dict[str, np.ndarray]]
        True event statuses (0=censored/baseline, 1=event, 2=competing).
    times : Union[np.ndarray, Dict[str, np.ndarray]]
        Event or censoring times.
    fixed_time_horizons : Sequence[float]
        Time horizons at which performance is evaluated.
    heuristics_sets : Union[Sequence[Dict[str, str]], None], optional
        Heuristics for handling censoring and competing events.
    stratified_by : Sequence[str], optional
        Sequence containing exactly one stratification key, either
        ``("probability_threshold",)`` or ``("ppcr",)``.
    by : float, optional
        Step size for grid generation. Defaults to ``0.01``.

    Returns
    -------
    _PredictionDistributionTimesData
        TypedDict containing ``bins``, ``operating_points``, ``rank_bins``, and
        ``cutoff_region_aj`` Polars DataFrames.
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

    if heuristics_sets is None:
        heuristics_sets = [
            {
                "censoring_heuristic": "adjusted",
                "competing_heuristic": "adjusted_as_negative",
            }
        ]

    _validate_time_input_alignment(probs=probs, reals=reals, times=times)

    evaluation_metadata_by_group = _build_evaluation_metadata(probs, reals, times)
    evaluation_ids = [
        metadata.evaluation for metadata in evaluation_metadata_by_group.values()
    ]
    if len(evaluation_ids) != len(set(evaluation_ids)):
        raise ValueError("Duplicate evaluation identifiers detected.")

    evaluation_keys = list(evaluation_metadata_by_group.keys())

    # Generate authoritative binned classification data and performance metrics
    binned_adj = prepare_binned_classification_data_times(
        probs=probs,
        reals=reals,
        times=times,
        fixed_time_horizons=list(fixed_time_horizons),
        heuristics_sets=list(heuristics_sets),
        stratified_by=stratified_by,
        by=by,
        risk_set_scope=["pooled_by_cutoff"],
    )

    performance_data = prepare_performance_data_times(
        probs=probs,
        reals=reals,
        times=times,
        fixed_time_horizons=list(fixed_time_horizons),
        heuristics_sets=list(heuristics_sets),
        stratified_by=stratified_by,
        by=by,
    )

    # Filter to pooled_by_cutoff rows for region-level cutoff AJ estimates
    pooled = binned_adj.filter(pl.col("risk_set_scope") == "pooled_by_cutoff")

    pred_label_dtype = pooled.schema["prediction_label"]

    # Construct complete grid for all cutoff x prediction_label pairs (2 per cutoff)
    grid = pooled.select(
        [
            "reference_group",
            "fixed_time_horizon",
            "censoring_heuristic",
            "competing_heuristic",
            "stratified_by",
            "chosen_cutoff",
        ]
    ).unique()

    region_labels_df = pl.DataFrame(
        {
            "prediction_label": pl.Series(
                ["predicted_positives", "predicted_negatives"],
                dtype=pred_label_dtype,
            )
        }
    )
    full_aj_grid = grid.join(region_labels_df, how="cross")

    state_masses = (
        pooled.group_by(
            [
                "reference_group",
                "fixed_time_horizon",
                "censoring_heuristic",
                "competing_heuristic",
                "stratified_by",
                "chosen_cutoff",
                "prediction_label",
                "reals_labels",
            ]
        )
        .agg(pl.col("reals_estimate").sum())
        .pivot(on="reals_labels", values="reals_estimate")
        .fill_null(0.0)
    )

    outcome_masses = (
        pooled.group_by(
            [
                "reference_group",
                "fixed_time_horizon",
                "censoring_heuristic",
                "competing_heuristic",
                "stratified_by",
                "chosen_cutoff",
                "prediction_label",
                "classification_outcome",
            ]
        )
        .agg(pl.col("reals_estimate").sum())
        .pivot(on="classification_outcome", values="reals_estimate")
        .fill_null(0.0)
    )

    joined_aj = (
        full_aj_grid.join(
            state_masses,
            on=[
                "reference_group",
                "fixed_time_horizon",
                "censoring_heuristic",
                "competing_heuristic",
                "stratified_by",
                "chosen_cutoff",
                "prediction_label",
            ],
            how="left",
        )
        .join(
            outcome_masses,
            on=[
                "reference_group",
                "fixed_time_horizon",
                "censoring_heuristic",
                "competing_heuristic",
                "stratified_by",
                "chosen_cutoff",
                "prediction_label",
            ],
            how="left",
        )
        .fill_null(0.0)
    )

    # Ensure all required state and outcome mass columns exist
    required_cols = [
        "real_positives",
        "real_negatives",
        "real_competing",
        "real_censored",
        "true_positives",
        "false_positives",
        "true_negatives",
        "false_negatives",
    ]
    for col in required_cols:
        if col not in joined_aj.columns:
            joined_aj = joined_aj.with_columns(pl.lit(0.0).alias(col))

    joined_aj = joined_aj.rename(
        {
            "real_positives": "real_positives_est",
            "real_negatives": "real_negatives_est",
            "real_competing": "real_competing_est",
            "real_censored": "real_censored_est",
        }
    )

    # Map reference_group to evaluation metadata columns
    eval_rows = []
    for ref_group, metadata in evaluation_metadata_by_group.items():
        eval_rows.append(
            {
                "reference_group": ref_group,
                "evaluation": metadata.evaluation,
                "model": metadata.model,
                "population": metadata.population,
            }
        )
    eval_meta_df = pl.DataFrame(eval_rows)

    # Cast reference_group column in eval_meta_df to match joined_aj
    ref_dtype = joined_aj.schema["reference_group"]
    eval_meta_df = eval_meta_df.with_columns(pl.col("reference_group").cast(ref_dtype))

    cutoff_region_aj = (
        joined_aj.join(eval_meta_df, on="reference_group", how="left")
        .drop("reference_group")
        .with_columns(pl.col("prediction_label").cast(pl.String))
        .select(
            [
                "evaluation",
                "model",
                "population",
                "fixed_time_horizon",
                "censoring_heuristic",
                "competing_heuristic",
                "stratified_by",
                "chosen_cutoff",
                "prediction_label",
                "real_positives_est",
                "real_negatives_est",
                "real_competing_est",
                "real_censored_est",
                "true_positives",
                "false_positives",
                "true_negatives",
                "false_negatives",
            ]
        )
    )

    # Build operating_points DataFrame from performance_data and evaluation_metadata
    perf_meta_df = eval_meta_df.with_columns(
        pl.col("reference_group").cast(performance_data.schema["reference_group"])
    )
    operating_points = (
        performance_data.join(perf_meta_df, on="reference_group", how="left")
        .drop("reference_group")
        .select(
            [
                "evaluation",
                "model",
                "population",
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
        )
    )

    # Build raw histogram bins and rank_bins
    eval_bins_frames = []
    eval_rank_bins_frames = []

    for evaluation_key in evaluation_keys:
        evaluation_metadata = evaluation_metadata_by_group[evaluation_key]

        probabilities = np.asarray(probs[evaluation_key], dtype=float)
        if isinstance(reals, dict):
            outcomes = np.asarray(reals[evaluation_key], dtype=int)
        else:
            outcomes = np.asarray(reals, dtype=int)

        # In both probability_threshold and ppcr modes, bins should be generated on genuine raw probability boundaries
        interval_boundaries = np.arange(0.0, 1.0 + float(by), float(by))
        interval_boundaries = np.clip(interval_boundaries, 0.0, 1.0)
        interval_boundaries = np.unique(interval_boundaries)

        eval_bins = _aggregate_bins_for_evaluation_times(
            probabilities=probabilities,
            outcomes=outcomes,
            interval_boundaries=interval_boundaries,
            evaluation_metadata=evaluation_metadata,
        )
        eval_bins_frames.append(eval_bins)

        eval_rank_bins = _aggregate_rank_bins_for_evaluation_times(
            probabilities=probabilities,
            outcomes=outcomes,
            by=by,
            evaluation_metadata=evaluation_metadata,
        )
        eval_rank_bins_frames.append(eval_rank_bins)

    bins = pl.concat(eval_bins_frames, how="vertical")
    rank_bins = pl.concat(eval_rank_bins_frames, how="vertical")

    return _PredictionDistributionTimesData(
        bins=bins,
        operating_points=operating_points,
        rank_bins=rank_bins,
        cutoff_region_aj=cutoff_region_aj,
    )

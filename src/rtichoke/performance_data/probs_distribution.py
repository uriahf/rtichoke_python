"""Private producer for prediction distribution data reproducing static binary semantics."""

from typing import Dict, Sequence, TypedDict, Union
import numpy as np
import polars as pl

from rtichoke.performance_data.performance_data import (
    _validate_and_align_binary_inputs,
    prepare_performance_data,
)
from rtichoke.processing.evaluation_semantics import _build_evaluation_metadata


class _PredictionDistributionData(TypedDict):
    bins: pl.DataFrame
    operating_points: pl.DataFrame


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
        TypedDict containing ``bins`` and ``operating_points`` Polars DataFrames.
    """
    if not isinstance(stratified_by, (list, tuple)) or len(stratified_by) != 1:
        raise ValueError(
            "`stratified_by` must be a sequence containing exactly one element: "
            "'probability_threshold' or 'ppcr'."
        )

    strat_type = stratified_by[0]
    if strat_type not in ("probability_threshold", "ppcr"):
        raise ValueError(
            f"Unsupported stratification key {strat_type!r}. "
            "Must be 'probability_threshold' or 'ppcr'."
        )

    aligned_reals = _validate_and_align_binary_inputs(probs=probs, reals=reals)

    # Derive evaluation metadata
    dummy_times = np.array([])
    eval_metadata_map = _build_evaluation_metadata(probs, aligned_reals, dummy_times)

    evaluations = list(eval_metadata_map.keys())
    if len(evaluations) != len(set(evaluations)):
        raise ValueError("Duplicate evaluation identifiers detected.")

    # Call authoritative production performance data
    perf_df = prepare_performance_data(
        probs=probs,
        reals=aligned_reals,
        stratified_by=stratified_by,
        by=by,
    )

    # Dtypes for output DataFrames
    bins_schema = {
        "evaluation": pl.String,
        "model": pl.String,
        "population": pl.String,
        "lower": pl.Float64,
        "upper": pl.Float64,
        "include_lower": pl.Boolean,
        "include_upper": pl.Boolean,
        "n_positive": pl.UInt32,
        "n_negative": pl.UInt32,
    }

    op_schema = {
        "evaluation": pl.String,
        "model": pl.String,
        "population": pl.String,
        "type": pl.String,
        "value": pl.Float64,
        "cutoff": pl.Float64,
        "realized_ppcr": pl.Float64,
    }

    bins_rows = []
    op_rows = []

    for eval_key in evaluations:
        meta = eval_metadata_map[eval_key]

        eval_perf = perf_df.filter(pl.col("reference_group") == eval_key)

        p_vec = np.asarray(probs[eval_key], dtype=float)
        if isinstance(aligned_reals, dict):
            r_vec = np.asarray(aligned_reals[eval_key], dtype=int)
        else:
            r_vec = np.asarray(aligned_reals, dtype=int)

        # Build operating points rows
        for row in eval_perf.iter_rows(named=True):
            requested_val = float(
                row["ppcr"] if strat_type == "ppcr" else row["chosen_cutoff"]
            )
            effective_cutoff = float(row["chosen_cutoff"])
            n_obs = int(row["n"])
            pred_pos = int(row["predicted_positives"])
            realized_ppcr = float(pred_pos / n_obs) if n_obs > 0 else 0.0

            op_rows.append(
                {
                    "evaluation": meta.evaluation,
                    "model": meta.model,
                    "population": meta.population,
                    "type": strat_type,
                    "value": requested_val,
                    "cutoff": effective_cutoff,
                    "realized_ppcr": realized_ppcr,
                }
            )

        # Build interval boundaries from effective cutoffs
        cutoffs = eval_perf["chosen_cutoff"].to_numpy().astype(float)
        unique_bounds = np.unique(np.concatenate(([0.0, 1.0], cutoffs)))
        unique_bounds.sort()

        # Build interval specs: [0, 0] then (bounds[i], bounds[i+1]]
        intervals = [(0.0, 0.0, True, True)]
        if len(unique_bounds) > 1:
            for i in range(len(unique_bounds) - 1):
                intervals.append(
                    (float(unique_bounds[i]), float(unique_bounds[i + 1]), False, True)
                )

        # Vectorized assignment of observations to intervals
        is_zero = p_vec == 0.0
        pos_zero = int(np.sum(r_vec[is_zero] == 1))
        neg_zero = int(np.sum(r_vec[is_zero] == 0))

        bins_rows.append(
            {
                "evaluation": meta.evaluation,
                "model": meta.model,
                "population": meta.population,
                "lower": 0.0,
                "upper": 0.0,
                "include_lower": True,
                "include_upper": True,
                "n_positive": pos_zero,
                "n_negative": neg_zero,
            }
        )

        non_zero_mask = p_vec > 0.0
        p_nonzero = p_vec[non_zero_mask]
        r_nonzero = r_vec[non_zero_mask]

        if len(p_nonzero) > 0 and len(unique_bounds) > 1:
            # Bucket index for p_nonzero into (unique_bounds[i], unique_bounds[i+1]]
            # np.digitize(p, bounds, right=True) maps p in (bounds[i-1], bounds[i]] to i
            b_indices = np.digitize(p_nonzero, unique_bounds, right=True)

            # Accumulate counts per interval index i (1 <= i < len(unique_bounds))
            for i in range(1, len(unique_bounds)):
                in_bin = b_indices == i
                pos_count = int(np.sum(r_nonzero[in_bin] == 1))
                neg_count = int(np.sum(r_nonzero[in_bin] == 0))

                bins_rows.append(
                    {
                        "evaluation": meta.evaluation,
                        "model": meta.model,
                        "population": meta.population,
                        "lower": float(unique_bounds[i - 1]),
                        "upper": float(unique_bounds[i]),
                        "include_lower": False,
                        "include_upper": True,
                        "n_positive": pos_count,
                        "n_negative": neg_count,
                    }
                )
        else:
            for i in range(1, len(unique_bounds)):
                bins_rows.append(
                    {
                        "evaluation": meta.evaluation,
                        "model": meta.model,
                        "population": meta.population,
                        "lower": float(unique_bounds[i - 1]),
                        "upper": float(unique_bounds[i]),
                        "include_lower": False,
                        "include_upper": True,
                        "n_positive": 0,
                        "n_negative": 0,
                    }
                )

    bins_df = pl.DataFrame(bins_rows, schema=bins_schema)
    op_df = pl.DataFrame(op_rows, schema=op_schema)

    return _PredictionDistributionData(
        bins=bins_df,
        operating_points=op_df,
    )

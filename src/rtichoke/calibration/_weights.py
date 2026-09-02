"""Private outcome-weighted calibration preparation helpers."""

from __future__ import annotations

import numpy as np
import polars as pl

from .calibration import _make_deciles_dat_binary


def _prepare_calibration_bins(
    probs: np.ndarray,
    reals: np.ndarray,
    outcome_weights: np.ndarray | None = None,
    n_bins: int = 10,
) -> pl.DataFrame:
    """Prepare vector-level calibration bins with optional outcome weights.

    Prediction-bin membership and displayed mean predicted risk are always
    defined from the target-population predictions. ``outcome_weights`` affect
    only the observed outcome estimate inside each bin. This separation is
    intentional for later counterfactual calibration, where treatment/IPW
    weights identify the observed risk under an intervention but should not
    redefine the target population's prediction distribution.

    With no weights, this delegates to the established factual calibration
    helper so existing semantics remain unchanged.
    """
    p = np.asarray(probs).ravel()
    y = np.asarray(reals).ravel()

    if p.shape[0] != y.shape[0]:
        raise ValueError("probs and reals must have the same length.")
    if not isinstance(n_bins, int) or isinstance(n_bins, bool) or n_bins < 1:
        raise ValueError("n_bins must be a positive integer.")

    if outcome_weights is None:
        return _make_deciles_dat_binary({"model": p}, y, n_bins=n_bins).with_columns(
            pl.col("n").cast(pl.Float64).alias("outcome_weight_sum")
        )

    weights = np.asarray(outcome_weights, dtype=float).ravel()
    if weights.shape[0] != y.shape[0]:
        raise ValueError("outcome_weights must have the same length as reals.")
    if not np.all(np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError(
            "outcome_weights must contain only finite, non-negative values."
        )

    df = pl.DataFrame(
        {
            "reference_group": ["model"] * p.shape[0],
            "model": ["model"] * p.shape[0],
            "prob": p.astype(float, copy=False),
            "real": y.astype(float, copy=False),
            "outcome_weight": weights,
        }
    ).with_columns(
        ((pl.col("prob").rank("ordinal") - 1) * n_bins // pl.len() + 1).alias(
            "decile"
        )
    )

    bins = (
        df.group_by(["reference_group", "model", "decile"])
        .agg(
            pl.len().alias("n"),
            pl.mean("prob").alias("x"),
            pl.sum("real").alias("n_reals"),
            pl.sum("outcome_weight").alias("outcome_weight_sum"),
            (pl.col("outcome_weight") * pl.col("real"))
            .sum()
            .alias("weighted_sum_reals"),
        )
        .sort(["reference_group", "model", "decile"])
    )

    if bins.filter(pl.col("outcome_weight_sum") <= 0).height:
        raise ValueError(
            "Every calibration bin must have positive total outcome_weights."
        )

    return bins.with_columns(
        (pl.col("weighted_sum_reals") / pl.col("outcome_weight_sum")).alias("y")
    ).select(
        "reference_group",
        "model",
        "decile",
        "n",
        "x",
        "y",
        "n_reals",
        "outcome_weight_sum",
        "weighted_sum_reals",
    )

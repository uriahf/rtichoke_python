"""Public calibration API with a consistent population data contract."""

from typing import Dict, List, Union

import numpy as np
import polars as pl
from plotly.graph_objs._figure import Figure

from .calibration import (
    _add_hover_text_to_calibration_data,
    _apply_heuristics_and_censoring,
    _build_initial_df_for_times,
    _check_performance_type_by_probs_and_reals,
    _create_colors_dictionary_for_calibration,
    _create_plotly_curve_from_calibration_curve_list,
    _create_plotly_curve_from_calibration_curve_list_times,
    _create_reference_data_for_calibration_curve,
    _define_limits_for_calibration_plot,
)

_DEFAULT_COLORS = [
    "#1b9e77",
    "#d95f02",
    "#7570b3",
    "#e7298a",
    "#07004D",
    "#E6AB02",
    "#FE5F55",
    "#54494B",
    "#006E90",
    "#BC96E6",
    "#52050A",
    "#1F271B",
    "#BE7C4D",
    "#63768D",
    "#08A045",
    "#320A28",
    "#82FF9E",
    "#2176FF",
    "#D1603D",
    "#585123",
]


def _build_binary_calibration_df(
    probs: Dict[str, np.ndarray],
    reals: Union[np.ndarray, Dict[str, np.ndarray]],
) -> pl.DataFrame:
    """Normalize binary calibration inputs to one long data frame."""
    frames: list[pl.DataFrame] = []

    if isinstance(reals, dict):
        if probs.keys() == reals.keys():
            for population in reals:
                p = np.asarray(probs[population]).ravel()
                y = np.asarray(reals[population]).ravel()
                if p.shape[0] != y.shape[0]:
                    raise ValueError(
                        f"Length mismatch for population '{population}': "
                        f"probs has length {p.shape[0]} but reals has length {y.shape[0]}."
                    )
                frames.append(
                    pl.DataFrame(
                        {
                            "reference_group": population,
                            "model": population,
                            "prob": p.astype(float, copy=False),
                            "real": y.astype(float, copy=False),
                        }
                    )
                )
        elif len(probs) == 1:
            model, p_all = next(iter(probs.items()))
            p_all = np.asarray(p_all).ravel()
            populations = list(reals.keys())
            outcomes = [np.asarray(reals[population]).ravel() for population in populations]
            lengths = [len(y) for y in outcomes]
            n_total = sum(lengths)
            if p_all.shape[0] != n_total:
                raise ValueError(
                    f"probs['{model}'] length={p_all.shape[0]} does not match "
                    f"sum of population sizes={n_total}."
                )

            start = 0
            for population, y in zip(populations, outcomes):
                end = start + len(y)
                frames.append(
                    pl.DataFrame(
                        {
                            "reference_group": population,
                            "model": model,
                            "prob": p_all[start:end].astype(float, copy=False),
                            "real": y.astype(float, copy=False),
                        }
                    )
                )
                start = end
        else:
            raise ValueError(
                "When probs and reals are dictionaries, use matching population keys, "
                "or provide one concatenated probability vector for all populations."
            )
    else:
        y = np.asarray(reals).ravel()
        for model, prob_array in probs.items():
            p = np.asarray(prob_array).ravel()
            if p.shape[0] != y.shape[0]:
                raise ValueError(
                    f"probs['{model}'] length={p.shape[0]} does not match "
                    f"reals length={y.shape[0]}."
                )
            frames.append(
                pl.DataFrame(
                    {
                        "reference_group": model,
                        "model": model,
                        "prob": p.astype(float, copy=False),
                        "real": y.astype(float, copy=False),
                    }
                )
            )

    return pl.concat(frames, how="vertical")


def _make_deciles_from_df(df: pl.DataFrame, n_bins: int = 10) -> pl.DataFrame:
    prepared = df.with_columns(
        [
            pl.col("prob").cast(pl.Float64),
            pl.col("real").cast(pl.Float64),
            (
                (pl.col("prob").rank("ordinal").over(["reference_group", "model"]) - 1)
                * n_bins
                // pl.len().over(["reference_group", "model"])
                + 1
            ).alias("decile"),
        ]
    )
    return (
        prepared.group_by(["reference_group", "model", "decile"])
        .agg(
            [
                pl.len().alias("n"),
                pl.mean("prob").alias("x"),
                pl.mean("real").alias("y"),
                pl.sum("real").alias("n_reals"),
            ]
        )
        .sort(["reference_group", "model", "decile"])
    )


def _make_smooth_from_df(df: pl.DataFrame) -> pl.DataFrame:
    from statsmodels.nonparametric.smoothers_lowess import lowess

    frames: list[pl.DataFrame] = []
    for group in df.partition_by(["reference_group", "model"], maintain_order=True):
        reference_group = str(group["reference_group"][0])
        p = group["prob"].to_numpy()
        y = group["real"].to_numpy()
        if len(np.unique(p)) == 1:
            frames.append(
                pl.DataFrame(
                    {
                        "x": [float(p[0])],
                        "y": [float(np.mean(y))],
                        "reference_group": [reference_group],
                    }
                )
            )
            continue

        smoothed = lowess(y, p, it=0)
        xout = np.linspace(0, 1, 101)
        yout = np.interp(xout, smoothed[:, 0], smoothed[:, 1])
        frames.append(
            pl.DataFrame(
                {
                    "x": xout,
                    "y": yout,
                    "reference_group": [reference_group] * len(xout),
                }
            )
        )

    return pl.concat(frames, how="vertical")


def _make_histogram_from_df(df: pl.DataFrame) -> pl.DataFrame:
    frames: list[pl.DataFrame] = []
    for group in df.partition_by("reference_group", maintain_order=True):
        reference_group = str(group["reference_group"][0])
        counts, mids = np.histogram(
            group["prob"].to_numpy(), bins=np.arange(0, 1.01, 0.01)
        )
        hist = pl.DataFrame(
            {
                "mids": mids[:-1] + 0.005,
                "counts": counts,
                "reference_group": reference_group,
            }
        ).with_columns(
            (
                pl.col("counts").cast(str)
                + " observations in ["
                + (pl.col("mids") - 0.005).round(3).cast(str)
                + ", "
                + (pl.col("mids") + 0.005).round(3).cast(str)
                + "]"
            ).alias("text")
        )
        frames.append(hist)
    return pl.concat(frames, how="vertical")


def _reference_groups(df: pl.DataFrame) -> list[str]:
    return [str(value) for value in df["reference_group"].unique(maintain_order=True)]


def create_calibration_curve(
    probs: Dict[str, np.ndarray],
    reals: Union[np.ndarray, Dict[str, np.ndarray]],
    calibration_type: str = "discrete",
    size: int = 600,
    color_values: List[str] = _DEFAULT_COLORS,
) -> Figure:
    """Create a calibration curve.

    Parameters
    ----------
    probs : dict[str, numpy.ndarray]
        Predicted probabilities. When ``reals`` is a dictionary with matching keys,
        each key is treated as an independent population and may have its own sample size.
    reals : numpy.ndarray or dict[str, numpy.ndarray]
        Observed binary outcomes. Matching dictionary keys are paired population-by-population.
    calibration_type : str, default="discrete"
        Calibration rendering type, either ``"discrete"`` or ``"smooth"``.
    size : int, default=600
        Figure width and height in pixels.
    color_values : list[str]
        Colors used for population or model traces.

    Returns
    -------
    plotly.graph_objs.Figure
        Interactive calibration figure.
    """
    df = _build_binary_calibration_df(probs, reals)
    performance_type = _check_performance_type_by_probs_and_reals(probs, reals)
    deciles = _make_deciles_from_df(df)
    smooth = _make_smooth_from_df(df)
    deciles, smooth = _add_hover_text_to_calibration_data(
        deciles, smooth, performance_type
    )
    groups = _reference_groups(df)
    colors = _create_colors_dictionary_for_calibration(
        groups, color_values, performance_type
    )
    limits = _define_limits_for_calibration_plot(deciles)

    curve_data = {
        "deciles_dat": deciles,
        "smooth_dat": smooth,
        "reference_data": _create_reference_data_for_calibration_curve(),
        "histogram_for_calibration": _make_histogram_from_df(df),
        "axes_ranges": {"xaxis": limits, "yaxis": limits},
        "colors_dictionary": colors,
        "performance_type": [performance_type],
        "size": [(size, size)],
    }
    return _create_plotly_curve_from_calibration_curve_list(
        curve_data, calibration_type=calibration_type
    )


def create_calibration_curve_times(
    probs: Dict[str, np.ndarray],
    reals: Union[np.ndarray, Dict[str, np.ndarray]],
    times: Union[np.ndarray, Dict[str, np.ndarray]],
    fixed_time_horizons: List[float],
    heuristics_sets: List[Dict[str, str]],
    calibration_type: str = "discrete",
    size: int = 600,
    color_values: List[str] = _DEFAULT_COLORS,
) -> Figure:
    """Create time-dependent calibration curves.

    Parameters
    ----------
    probs : dict[str, numpy.ndarray]
        Predicted probabilities. Matching dictionary keys identify populations.
    reals : numpy.ndarray or dict[str, numpy.ndarray]
        Observed event indicators.
    times : numpy.ndarray or dict[str, numpy.ndarray]
        Observed event or censoring times.
    fixed_time_horizons : list[float]
        Time horizons to display.
    heuristics_sets : list[dict[str, str]]
        Censoring and competing-risk heuristic combinations.
    calibration_type : str, default="discrete"
        Calibration rendering type.
    size : int, default=600
        Figure width and height in pixels.
    color_values : list[str]
        Colors used for population or model traces.

    Returns
    -------
    plotly.graph_objs.Figure
        Interactive time-dependent calibration figure.
    """
    initial_df = _build_initial_df_for_times(probs, reals, times)
    performance_type = _check_performance_type_by_probs_and_reals(probs, reals)
    reference_groups = _reference_groups(initial_df)

    all_deciles: list[pl.DataFrame] = []
    all_smooth: list[pl.DataFrame] = []
    all_histograms: list[pl.DataFrame] = []

    for horizon in fixed_time_horizons:
        for heuristics in heuristics_sets:
            censoring_heuristic = heuristics["censoring_heuristic"]
            competing_heuristic = heuristics["competing_heuristic"]
            if (
                censoring_heuristic == "adjusted"
                or competing_heuristic == "adjusted_as_censored"
            ):
                continue

            adjusted = _apply_heuristics_and_censoring(
                initial_df, horizon, censoring_heuristic, competing_heuristic
            )
            if adjusted.height == 0:
                continue

            all_deciles.append(
                _make_deciles_from_df(adjusted).with_columns(
                    pl.lit(horizon).alias("fixed_time_horizon")
                )
            )
            all_smooth.append(
                _make_smooth_from_df(adjusted).with_columns(
                    pl.lit(horizon).alias("fixed_time_horizon")
                )
            )
            all_histograms.append(
                _make_histogram_from_df(adjusted).with_columns(
                    pl.lit(horizon).alias("fixed_time_horizon")
                )
            )

    if not all_deciles:
        raise ValueError("No data remaining after applying heuristics and time horizons.")

    deciles = pl.concat(all_deciles)
    smooth = pl.concat(all_smooth)
    histograms = pl.concat(all_histograms)
    deciles, smooth = _add_hover_text_to_calibration_data(
        deciles, smooth, performance_type
    )
    colors = _create_colors_dictionary_for_calibration(
        reference_groups, color_values, performance_type
    )
    limits = _define_limits_for_calibration_plot(deciles)

    curve_data = {
        "deciles_dat": deciles,
        "smooth_dat": smooth,
        "reference_data": _create_reference_data_for_calibration_curve(),
        "histogram_for_calibration": histograms,
        "axes_ranges": {"xaxis": limits, "yaxis": limits},
        "colors_dictionary": colors,
        "performance_type": [performance_type],
        "size": [(size, size)],
        "fixed_time_horizons": fixed_time_horizons,
        "reference_group_keys": reference_groups,
    }
    return _create_plotly_curve_from_calibration_curve_list_times(
        curve_data, calibration_type=calibration_type
    )

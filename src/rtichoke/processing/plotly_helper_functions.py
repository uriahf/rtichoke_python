"""
A module for helpers related to plotly
"""

import plotly.graph_objects as go
import polars as pl
import math
from typing import Any, Dict, Union, Sequence, cast
import numpy as np
from rtichoke.performance_data.performance_data import prepare_performance_data
from rtichoke.performance_data.performance_data_times import (
    prepare_performance_data_times,
)

_HOVER_LABELS = {
    "false_positive_rate": "1 - Specificity (FPR)",
    "sensitivity": "Sensitivity",
    "specificity": "Specificity",
    "lift": "Lift",
    "ppv": "PPV",
    "npv": "NPV",
    "net_benefit": "NB",
    "net_benefit_interventions_avoided": "Interventions Avoided (per 100)",
    "chosen_cutoff": "Prob. Threshold",
    "ppcr": "Predicted Positives",
}

DEFAULT_MODEBAR_BUTTONS_TO_REMOVE = [
    "zoom2d",
    "pan2d",
    "select2d",
    "lasso2d",
    "zoomIn2d",
    "zoomOut2d",
    "autoScale2d",
    "resetScale2d",
    "hoverClosestCartesian",
    "hoverCompareCartesian",
    "toggleSpikelines",
    "toImage",
]


def _create_rtichoke_plotly_curve_binary(
    probs: Dict[str, np.ndarray],
    reals: Union[np.ndarray, Dict[str, np.ndarray]],
    by: float = 0.01,
    stratified_by: Sequence[str] = ["probability_threshold"],
    size: int = 600,
    color_values=None,
    curve: str = "roc",
    min_p_threshold: float = 0,
    max_p_threshold: float = 1,
) -> go.Figure:
    performance_data = prepare_performance_data(
        probs=probs,
        reals=reals,
        stratified_by=stratified_by,
        by=by,
    )

    fig = _plot_rtichoke_curve_binary(
        performance_data=performance_data,
        stratified_by=stratified_by[0],
        curve=curve,
        size=size,
    )

    return fig


def _create_rtichoke_plotly_curve_times(
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
    min_p_threshold: float = 0,
    max_p_threshold: float = 1,
    by: float = 0.01,
    stratified_by: Sequence[str] = ["probability_threshold"],
    size: int = 600,
    color_values=None,
    curve: str = "roc",
) -> go.Figure:
    performance_data = prepare_performance_data_times(
        probs,
        reals,
        times,
        by=by,
        fixed_time_horizons=fixed_time_horizons,
        heuristics_sets=heuristics_sets,
        stratified_by=stratified_by,
    )

    rtichoke_curve_list_times = _create_rtichoke_curve_list_times(
        performance_data, stratified_by=stratified_by[0], curve=curve
    )

    fig = _create_plotly_curve_times(rtichoke_curve_list_times)

    return fig


def _plot_rtichoke_curve_binary(
    performance_data: pl.DataFrame,
    stratified_by: str = "probability_threshold",
    curve: str = "roc",
    size: int = 600,
    min_p_threshold: float = 0,
    max_p_threshold: float = 1,
) -> go.Figure:
    rtichoke_curve_list = _create_rtichoke_curve_list_binary(
        performance_data=performance_data,
        stratified_by=stratified_by,
        curve=curve,
        size=size,
        min_p_threshold=min_p_threshold,
        max_p_threshold=max_p_threshold,
    )

    fig = _create_plotly_curve_binary(rtichoke_curve_list)

    return fig


def _grid(start: float, stop: float, step: float) -> pl.Series:
    """Like R seq(start, stop, by=step)."""
    n = int(round((stop - start) / step)) + 1
    return pl.Series(np.round(np.linspace(start, stop, n), 10))


def _is_several_populations(perf_dat_type: str) -> bool:
    return perf_dat_type.strip().lower() == "several populations"


# Perfect/strategy reference formulas (vectorized)
def _perfect_gains_y(x: pl.Series, p: float) -> pl.Series:
    # Gains perfect: y = min(x/p, 1); if p<=0 => 0
    xa = x.to_numpy()
    if p <= 0:
        return pl.Series(np.zeros_like(xa, dtype=float))
    return pl.Series(np.minimum(xa / p, 1.0))


def _perfect_lift_y_series(x: pl.Series, p: float) -> pl.Series:
    if not (0.0 <= p <= 1.0):
        raise ValueError(f"p must be in [0,1], got {p}")
    if p == 0.0:
        return pl.Series(np.full(x.len(), np.nan), dtype=pl.Float64)
    if p == 1.0:
        return pl.Series(np.full(x.len(), 1.0), dtype=pl.Float64)

    # materialize via DataFrame -> Series to avoid Expr return
    df = pl.DataFrame({"x": x.cast(pl.Float64)})
    y = df.select(
        pl.when(pl.col("x") <= p)
        .then(1.0 / p)
        .otherwise(1.0 / p + ((1.0 - 1.0 / p) / (1.0 - p)) * (pl.col("x") - p))
        .cast(pl.Float64)
        .alias("y")
    ).to_series()
    return y


def _perfect_lift_expr(x_col: str = "x", p_col: str = "p") -> pl.Expr:
    # piecewise: p<=0 -> NaN, p>=1 -> 1, x<=p -> 1/p, else linear to 1 at x=1
    x = pl.col(x_col)
    p = pl.col(p_col)
    m = (1.0 - 1.0 / p) / (1.0 - p)
    return (
        pl.when(p <= 0)
        .then(pl.lit(np.nan))
        .when(p >= 1)
        .then(pl.lit(1.0))
        .when(x <= p)
        .then(1.0 / p)
        .otherwise(1.0 / p + m * (x - p))
        .cast(pl.Float64)
    )


def _random_guess_pr_expr(p_col: str = "p", x_col: str = "x") -> pl.Expr:
    # Baseline precision equals prevalence p; undefined (NaN) if p <= 0
    p = pl.col(p_col)
    return pl.when(p <= 0).then(pl.lit(np.nan)).otherwise(p).cast(pl.Float64)


def _treat_all_nb_y(x: pl.Series, p: float) -> pl.Series:
    # Decision curve "treat all" NB = p - (1-p)*x/(1-x)
    xa = x.to_numpy()
    y = p - (1 - p) * (xa / (1 - xa))
    return pl.Series(y)


def _treat_none_interventions_avoided_y(x: pl.Series, p: float) -> pl.Series:
    # Interventions avoided (per 100) for "treat none": 100*(1 - p - p*(1-x)/x)
    xa = x.to_numpy()
    y = 100.0 * (1 - p - p * (1 - xa) / xa)
    return pl.Series(y)


def _htext_pr(title_expr: pl.Expr) -> pl.Expr:
    return pl.format(
        "<b>{}</b><br>PPV: {}<br>Sensitivity: {}",
        title_expr,
        pl.col("y").round(3),
        pl.col("x").round(2),
    ).alias("text")


def _ensure_series_float(x) -> pl.Series:
    return (
        x.cast(pl.Float64)
        if isinstance(x, pl.Series)
        else pl.Series(x, dtype=pl.Float64)
    )


def _perfect_gains_expr(x_col: str = "x", p_col: str = "p") -> pl.Expr:
    x, p = pl.col(x_col), pl.col(p_col)
    return (
        pl.when(p == 0)
        .then(pl.lit(np.nan))  # no positives -> undefined
        .when(p == 1)
        .then(x)  # all positives -> y=x
        .when(x < p)
        .then(x / p)  # linear up to full recall at x=p
        .otherwise(pl.lit(1.0))  # plateau at 1 afterwards
        .cast(pl.Float64)
    )


def _perfect_gains_series(x: pl.Series, p: float) -> pl.Series:
    if p <= 0.0:
        return pl.Series(np.full(len(x), np.nan), dtype=pl.Float64)
    if p >= 1.0:
        return x.cast(pl.Float64)
    df = pl.DataFrame({"x": x})
    return df.select(
        pl.when(pl.col("x") <= p)
        .then(pl.col("x") / p)
        .otherwise(1.0)
        .cast(pl.Float64)
        .alias("y")
    ).to_series()


def _odds_expr(x_col: str = "x") -> pl.Expr:
    x = pl.col(x_col)
    return pl.when(x == 0).then(pl.lit("∞")).otherwise(((1 - x) / x).round(2))


def _treat_all_nb_expr(x_col: str = "x", p_col: str = "p") -> pl.Expr:
    # Net Benefit (treat-all): NB = p - (1 - p) * (pt / (1 - pt))
    # Guard x=1 (division by zero) and invalid p
    x, p = pl.col(x_col), pl.col(p_col)
    w = x / (1 - x)
    return (
        pl.when((p < 0) | (p > 1) | (x >= 1))
        .then(pl.lit(np.nan))
        .otherwise(p - (1 - p) * w)
        .cast(pl.Float64)
    )


def _htext_nb(title_expr: pl.Expr, x_col: str = "x") -> pl.Expr:
    return pl.format(
        "<b>{}</b><br>NB: {}<br>Probability Threshold: {}<br>Odds of Prob. Threshold: 1:{}",
        title_expr,
        pl.col("y").round(3),
        pl.col(x_col),
        _odds_expr(x_col),
    ).alias("text")


def _htext_ia(title_expr: pl.Expr, x_col: str = "x") -> pl.Expr:
    return pl.format(
        "<b>{}</b><br>Interventions Avoided (per 100): {}"
        "<br>Probability Threshold: {}<br>Odds of Prob. Threshold: 1:{}",
        title_expr,
        pl.col("y").round(3),
        pl.col(x_col),
        ((1 - pl.col(x_col)) / pl.col(x_col)).round(2),
    ).alias("text")


def _create_reference_lines_data(
    curve: str,
    aj_estimates_from_performance_data: pl.DataFrame,
    multiple_populations: bool,
    min_p_threshold: float = 0.0,
    max_p_threshold: float = 1.0,
) -> pl.DataFrame:
    curve = curve.strip().lower()
    # --- ROC ---
    if curve == "roc":
        x = _grid(0.0, 1.0, 0.01)
        return pl.DataFrame(
            {
                "reference_group": pl.Series(["random_guess"] * len(x)),
                "x": x,
                "y": x,
            }
        ).with_columns(
            pl.format(
                "<b>Random Guess</b><br>Sensitivity: {}<br>1 - Specificity: {}",
                pl.col("y"),
                pl.col("x"),
            ).alias("text")
        )

    # --- Lift ---
    if curve == "lift":
        x = _grid(0.01, 1.0, 0.01)
        x_s = pl.Series("x", x, dtype=pl.Float64)

        if multiple_populations:
            aj_df = aj_estimates_from_performance_data.select(
                pl.col("reference_group"),
                pl.col("aj_estimate").cast(pl.Float64).alias("p"),
            )

            # random-guess (y=1 unless all p==0 -> NaN)
            all_zero = (
                aj_df["p"].len() > 0
                and float(cast(float, aj_df["p"].max())) == 0.0
                and float(cast(float, aj_df["p"].min())) == 0.0
            )
            rand_y = pl.Series(
                np.full(len(x_s), np.nan) if all_zero else np.ones(len(x_s)),
                dtype=pl.Float64,
            )

            random_guess = pl.DataFrame(
                {"reference_group": ["random_guess"] * len(x_s), "x": x_s, "y": rand_y}
            ).with_columns(
                pl.format(
                    "<b>Random Guess</b><br>Lift: {}<br>Predicted Positives: {}%",
                    pl.col("y").round(3),
                    (100 * pl.col("x")).round(0),
                ).alias("text")
            )

            # perfect per population (cross-join x with per-group p)
            perfect_df = (
                pl.DataFrame({"x": x_s})
                .join(aj_df, how="cross")
                .with_columns(
                    _perfect_lift_expr("x", "p").alias("y"),
                    pl.format("perfect_model_{}", pl.col("reference_group")).alias(
                        "reference_group_fmt"
                    ),
                )
                .with_columns(
                    pl.format(
                        "<b>Perfect Prediction ({})</b><br>Lift: {}<br>Predicted Positives: {}%",
                        pl.col("reference_group"),
                        pl.col("y").round(3),
                        (100 * pl.col("x")).round(0),
                    ).alias("text")
                )
                .select(
                    [
                        pl.col("reference_group_fmt").alias("reference_group"),
                        "x",
                        "y",
                        "text",
                    ]
                )
            )

            return pl.concat([random_guess, perfect_df], how="vertical")

        else:
            # single population
            p = float(
                aj_estimates_from_performance_data.select(
                    pl.col("aj_estimate").cast(pl.Float64).first()
                ).item()
            )

            rand_y = (
                pl.Series(np.ones(len(x_s)), dtype=pl.Float64)
                if p > 0.0
                else pl.Series(np.full(len(x_s), np.nan), dtype=pl.Float64)
            )

            perfect_y = _perfect_lift_y_series(x_s, p)

        return pl.concat(
            [
                pl.DataFrame(
                    {
                        "reference_group": ["random_guess"] * len(x_s),
                        "x": x_s,
                        "y": rand_y,
                    }
                ).with_columns(
                    pl.format(
                        "<b>Random Guess</b><br>Lift: {}<br>Predicted Positives: {}%",
                        pl.col("y").round(3),
                        (100 * pl.col("x")).round(0),
                    ).alias("text")
                ),
                pl.DataFrame(
                    {
                        "reference_group": ["perfect_model"] * len(x_s),
                        "x": x_s,
                        "y": perfect_y,
                    }
                ).with_columns(
                    pl.format(
                        "<b>Perfect Prediction</b><br>Lift: {}<br>Predicted Positives: {}%",
                        pl.col("y").round(3),
                        (100 * pl.col("x")).round(0),
                    ).alias("text")
                ),
            ],
            how="vertical",
        )

    # --- Precision–Recall ---
    if curve == "precision recall":
        x = pl.Series("x", _grid(0.01, 1.0, 0.01), dtype=pl.Float64)

        def _htext(title_expr: pl.Expr) -> pl.Expr:
            return pl.format(
                "<b>{}</b><br>PPV: {}<br>Sensitivity: {}",
                title_expr,
                pl.col("y").round(3),
                pl.col("x").round(2),
            ).alias("text")

        if multiple_populations:
            # Expect aj_estimates_from_performance_data: [reference_group, aj_estimate]
            aj_df = aj_estimates_from_performance_data.select(
                pl.col("reference_group"),
                pl.col("aj_estimate").cast(pl.Float64).alias("p"),
            )

            base = pl.DataFrame({"x": x})

            # Random guess per population
            random_guess = (
                base.join(aj_df, how="cross")
                .with_columns(
                    _random_guess_pr_expr("p", "x").alias("y"),
                    pl.format("random_guess_{}", pl.col("reference_group")).alias(
                        "reference_group"
                    ),
                )
                .with_columns(
                    _htext(pl.format("Random Guess ({})", pl.col("reference_group")))
                )
                .select(["reference_group", "x", "y", "text"])
            )

            return random_guess

        else:
            # Single population
            p = float(
                aj_estimates_from_performance_data.select(
                    pl.col("aj_estimate").cast(pl.Float64).first()
                ).item()
            )

            n = len(x)
            y_baseline = (
                pl.Series(np.full(n, np.nan), dtype=pl.Float64)
                if p <= 0.0
                else pl.Series(np.full(n, p), dtype=pl.Float64)
            )

            return pl.DataFrame(
                {
                    "reference_group": ["random_guess"] * n,
                    "x": x,
                    "y": y_baseline,
                }
            ).with_columns(_htext(pl.lit("Random Guess")))

    # --- Gains ---
    if curve == "gains":
        x = pl.Series("x", _grid(0.0, 1.0, 0.01), dtype=pl.Float64)
        base = pl.DataFrame({"x": x})

        def _htext(title: pl.Expr) -> pl.Expr:
            return pl.format(
                "<b>{}</b><br>Sensitivity: {}<br>Predicted Positives: {}%",
                title,
                pl.col("y").round(3),
                (100 * pl.col("x")).round(0),
            ).alias("text")

        random_guess = (
            base.with_columns(
                pl.lit("random_guess").alias("reference_group"),
                pl.col("x").alias("y"),
            )
            .with_columns(_htext(pl.lit("Random Guess")))
            .select(["reference_group", "x", "y", "text"])
        )

        if multiple_populations:
            # Expect DF with columns: reference_group, aj_estimate
            aj_df = aj_estimates_from_performance_data.select(
                pl.col("reference_group"),
                pl.col("aj_estimate").cast(pl.Float64).alias("p"),
            )

            perfect_df = (
                base.join(aj_df, how="cross")
                .with_columns(
                    _perfect_gains_expr("x", "p").alias("y"),
                    pl.format("perfect_model_{}", pl.col("reference_group")).alias(
                        "reference_group"
                    ),
                )
                .with_columns(
                    _htext(
                        pl.format("Perfect Prediction ({})", pl.col("reference_group"))
                    )
                )
                .select(["reference_group", "x", "y", "text"])
            )

            return pl.concat([random_guess, perfect_df], how="vertical")

        else:
            # Single population: take first aj_estimate
            p = float(
                aj_estimates_from_performance_data.select(
                    pl.col("aj_estimate").cast(pl.Float64).first()
                ).item()
            )
            perfect_y = _perfect_gains_series(x, p)
            perfect_df = pl.DataFrame(
                {"reference_group": ["perfect_model"] * len(x), "x": x, "y": perfect_y}
            ).with_columns(_htext(pl.lit("Perfect Prediction")))
            return pl.concat([random_guess, perfect_df], how="vertical")

    # ===== Decision Curve =====
    if curve == "decision":
        x = pl.Series("x", _grid(0.0, 0.99, 0.01), dtype=pl.Float64)
        base = pl.DataFrame({"x": x})

        # Treat-none (reference line, NB=0)
        treat_none = (
            base.with_columns(
                pl.lit("treat_none").alias("reference_group"),
                pl.lit(0.0, dtype=pl.Float64).alias("y"),
            )
            .with_columns(_htext_nb(pl.lit("Treat None")))
            .select(["reference_group", "x", "y", "text"])
        )

        if multiple_populations:
            # expect aj_estimates_from_performance_data: [reference_group, aj_estimate]
            aj_df = aj_estimates_from_performance_data.select(
                pl.col("reference_group"),
                pl.col("aj_estimate").cast(pl.Float64).alias("p"),
            )

            treat_all = (
                base.join(aj_df, how="cross")
                .with_columns(
                    _treat_all_nb_expr("x", "p").alias("y"),
                    pl.format("treat_all_{}", pl.col("reference_group")).alias(
                        "reference_group"
                    ),
                )
                .with_columns(
                    _htext_nb(pl.format("Treat All ({})", pl.col("reference_group")))
                )
                .select(["reference_group", "x", "y", "text"])
            )

            df = pl.concat([treat_none, treat_all], how="vertical")

        else:
            # single population
            p = float(
                aj_estimates_from_performance_data.select(
                    pl.col("aj_estimate").cast(pl.Float64).first()
                ).item()
            )

            treat_all = (
                base.with_columns(
                    _treat_all_nb_expr("x", p_col=None).map_elements(
                        lambda v: v, return_dtype=pl.Float64
                    )  # no-op cast
                    if False
                    # (keeps linter happy; we inline p below)
                    else pl.when((p < 0) | (p > 1) | (pl.col("x") >= 1))
                    .then(pl.lit(np.nan))
                    .otherwise(
                        pl.lit(p) - (1 - pl.lit(p)) * (pl.col("x") / (1 - pl.col("x")))
                    )
                    .cast(pl.Float64)
                    .alias("y")
                )
                .with_columns(pl.lit("treat_all").alias("reference_group"))
                .with_columns(_htext_nb(pl.lit("Treat All")))
                .select(["reference_group", "x", "y", "text"])
            )

            df = pl.concat([treat_none, treat_all], how="vertical")

        # clamp thresholds post-build
        return df.filter(
            (pl.col("x") >= min_p_threshold) & (pl.col("x") <= max_p_threshold)
        )

    # ===== Interventions Avoided (reference lines) =====
    if curve == "interventions avoided":
        x = pl.Series(
            "x", _grid(0.01, 0.99, 0.01), dtype=pl.Float64
        )  # avoid x=0,1 divisions
        base = pl.DataFrame({"x": x})

        # Treat-all reference (0 per 100)
        treat_all_ref = (
            base.with_columns(
                pl.lit("treat_all").alias("reference_group"),
                pl.lit(0.0, dtype=pl.Float64).alias("y"),
            )
            .with_columns(_htext_ia(pl.lit("Treat All")))
            .select(["reference_group", "x", "y", "text"])
        )

        if multiple_populations:
            # expect aj_estimates_from_performance_data: [reference_group, aj_estimate]
            aj_df = aj_estimates_from_performance_data.select(
                pl.col("reference_group"),
                pl.col("aj_estimate").cast(pl.Float64).alias("p"),
            )

            # Use your existing helper for correctness of the IA math
            parts = [treat_all_ref]
            for row in aj_df.iter_rows(named=True):
                name, p = row["reference_group"], float(row["p"])
                y = _treat_none_interventions_avoided_y(x, p)  # your helper
                parts.append(
                    pl.DataFrame(
                        {
                            "reference_group": [f"treat_none_{name}"] * len(x),
                            "x": x,
                            "y": y,
                        }
                    ).with_columns(_htext_ia(pl.lit(f"Treat None ({name})")))
                )
            df = pl.concat(parts, how="vertical")

        else:
            p = float(
                aj_estimates_from_performance_data.select(
                    pl.col("aj_estimate").cast(pl.Float64).first()
                ).item()
            )
            y = _treat_none_interventions_avoided_y(x, p)  # your helper

            df = pl.concat(
                [
                    treat_all_ref,
                    pl.DataFrame(
                        {"reference_group": ["treat_none"] * len(x), "x": x, "y": y}
                    ).with_columns(_htext_ia(pl.lit("Treat None"))),
                ],
                how="vertical",
            )

        return df.filter(
            (pl.col("x") >= min_p_threshold) & (pl.col("x") <= max_p_threshold)
        )

    return pl.DataFrame(
        schema={
            "reference_group": pl.Utf8,
            "x": pl.Float64,
            "y": pl.Float64,
            "text": pl.Utf8,
        }
    )


def create_non_interactive_curve_polars(
    performance_data_ready_for_curve, reference_group_color, reference_group
):
    """

    Parameters
    ----------
    performance_data_ready_for_curve :

    reference_group_color :


    Returns
    -------

    """

    non_interactive_curve = go.Scatter(
        x=performance_data_ready_for_curve["x"],
        y=performance_data_ready_for_curve["y"],
        mode="markers+lines",
        hoverinfo="text",
        # hovertext=performance_data_ready_for_curve["text"],
        name=reference_group,
        legendgroup=reference_group,
        line={"width": 2, "color": reference_group_color},
    )
    return non_interactive_curve


def create_non_interactive_curve(
    performance_data_ready_for_curve, reference_group_color, reference_group
):
    """

    Parameters
    ----------
    performance_data_ready_for_curve :

    reference_group_color :


    Returns
    -------

    """
    performance_data_ready_for_curve = performance_data_ready_for_curve.dropna()

    non_interactive_curve = go.Scatter(
        x=performance_data_ready_for_curve["x"].values.tolist(),
        y=performance_data_ready_for_curve["y"].values.tolist(),
        mode="markers+lines",
        hoverinfo="text",
        hovertext=performance_data_ready_for_curve["text"].values.tolist(),
        name=reference_group,
        legendgroup=reference_group,
        line={"width": 2, "color": reference_group_color},
    )
    return non_interactive_curve


def create_interactive_marker(
    performance_data_ready_for_curve, interactive_marker_color, k, reference_group
):
    """

    Parameters
    ----------
    performance_data_ready_for_curve :

    reference_group_color :

    k :


    Returns
    -------

    """
    performance_data_ready_for_curve = performance_data_ready_for_curve.assign(
        column_name=performance_data_ready_for_curve.loc[:, "y"].fillna(-1)
    )

    interactive_marker = go.Scatter(
        x=[performance_data_ready_for_curve["x"].values.tolist()[k]],
        y=[performance_data_ready_for_curve["y"].values.tolist()[k]],
        mode="markers",
        hoverinfo="text",
        hovertext=[performance_data_ready_for_curve["text"].values.tolist()[k]],
        name=reference_group,
        legendgroup=reference_group,
        showlegend=False,
        marker={
            "size": 12,
            "color": interactive_marker_color,
            "line": {"width": 2, "color": "black"},
        },
    )
    return interactive_marker


def create_reference_lines_for_plotly(reference_data, reference_line_color):
    """Creates a plotly scatter object of the reference lines

    Parameters
    ----------
    reference_data :

    reference_line_color :


    Returns
    -------

    """
    reference_lines = go.Scatter(
        x=reference_data["x"].values.tolist(),
        y=reference_data["y"].values.tolist(),
        mode="lines",
        hoverinfo="text",
        hovertext=reference_data["text"].values.tolist(),
        name="reference_line",
        line={"width": 2, "color": reference_line_color, "dash": "dot"},
        showlegend=False,
    )
    return reference_lines


_CURVE_CONFIG = {
    "roc": (
        "false_positive_rate",
        "sensitivity",
        "1 - Specificity",
        "Sensitivity",
    ),
    "precision recall": (
        "sensitivity",
        "ppv",
        "Sensitivity",
        "PPV",
    ),
    "lift": (
        "ppcr",
        "lift",
        "Predicted Positives (Rate)",
        "Lift",
    ),
    "gains": (
        "ppcr",
        "sensitivity",
        "Predicted Positives (Rate)",
        "Sensitivity",
    ),
    "decision": (
        "chosen_cutoff",
        "net_benefit",
        "Probability Threshold",
        "Net Benefit",
    ),
    "interventions avoided": (
        "chosen_cutoff",
        "net_benefit_interventions_avoided",
        "Probability Threshold",
        "Interventions Avoided (per 100)",
    ),
}


def _finite_vals(series: pl.Series) -> list[float]:
    vals = series.to_list()
    out = []
    for v in vals:
        if v is None:
            continue
        # Allow ints too
        if (
            isinstance(v, (int, float))
            and math.isfinite(v)
            and not math.isnan(float(v))
        ):
            out.append(float(v))
    return out


def extract_axes_ranges(
    performance_data_ready: pl.DataFrame,
    curve: str,
    min_p_threshold: float = 0.0,
    max_p_threshold: float = 1.0,
) -> dict[str, list[float]]:
    y_vals = _finite_vals(performance_data_ready["y"])

    if curve == "roc":
        rng = {"xaxis": [0.0, 1.0], "yaxis": [0.0, 1.0]}

    elif curve == "precision recall":
        rng = {"xaxis": [0.0, 1.0], "yaxis": [0.0, 1.0]}

    elif curve == "gains":
        rng = {"xaxis": [0.0, 1.0], "yaxis": [0.0, 1.0]}

    elif curve == "lift":
        max_y = max([1.0] + y_vals) if y_vals else 1.0
        rng = {"xaxis": [0.0, 1.0], "yaxis": [0.0, max_y]}

    elif curve == "decision":
        max_y = max(y_vals) if y_vals else 0.0
        min_y = min(y_vals) if y_vals else 0.0
        rng = {
            "xaxis": [float(min_p_threshold), float(max_p_threshold)],
            "yaxis": [min(min_y, 0.0), max_y],
        }

    elif curve == "interventions avoided":
        min_y = min(y_vals) if y_vals else 0.0
        rng = {
            "xaxis": [float(min_p_threshold), float(max_p_threshold)],
            "yaxis": [min(0.0, min_y), 100.0],
        }

    else:
        # Sensible default
        rng = {"xaxis": [0.0, 1.0], "yaxis": [0.0, 1.0]}

    # Match the R post-step: purrr::map(~ extend_axis_ranges(.x))
    rng["xaxis"] = _extend_axis_ranges(rng["xaxis"])
    rng["yaxis"] = _extend_axis_ranges(rng["yaxis"])
    return rng


def _extend_axis_ranges(bounds, pad_frac=0.02):
    lo, hi = bounds
    # Handle None or identical values
    if lo is None or hi is None:
        return bounds
    span = hi - lo
    if span <= 0:
        pad = 1e-6
        return [lo - pad, hi + pad]
    pad = span * pad_frac
    return [lo - pad, hi + pad]


def _get_prevalence_from_performance_data(
    performance_data: pl.DataFrame,
) -> dict[str, float]:
    cols_to_keep = [
        c for c in ["model", "population", "ppv"] if c in performance_data.columns
    ]
    if "ppcr" not in performance_data.columns or "ppv" not in performance_data.columns:
        raise ValueError("performance_data must include 'ppcr' and 'PPV' columns")

    df = performance_data.filter(pl.col("ppcr") == 1).select(cols_to_keep).unique()

    if len(df.columns) == 1:
        return {"single": float(df["ppv"][0])}

    key_col = df.columns[0]
    return dict(zip(df[key_col].to_list(), df["ppv"].to_list()))


def _get_aj_estimates_from_performance_data(
    performance_data: pl.DataFrame,
) -> pl.DataFrame:
    return (
        performance_data.select("reference_group", "real_positives", "n")
        .unique()
        .with_columns((pl.col("real_positives") / pl.col("n")).alias("aj_estimate"))
        .select(pl.col("reference_group"), pl.col("aj_estimate"))
    )


def _get_aj_estimates_from_performance_data_times(
    performance_data: pl.DataFrame,
) -> pl.DataFrame:
    return (
        performance_data.filter(
            (pl.col("chosen_cutoff") == 0) | (pl.col("chosen_cutoff") == 1)
        )
        .select("reference_group", "fixed_time_horizon", "real_positives", "n")
        .unique()
        .with_columns((pl.col("real_positives") / pl.col("n")).alias("aj_estimate"))
        .select(
            pl.col("reference_group"),
            pl.col("fixed_time_horizon"),
            pl.col("aj_estimate"),
        )
        .sort(by=["reference_group", "fixed_time_horizon"])
    )


def _check_if_multiple_populations_are_being_validated_times(
    aj_estimates: pl.DataFrame,
) -> bool:
    max_val = (
        aj_estimates.group_by("fixed_time_horizon")
        .agg(pl.col("aj_estimate").n_unique().alias("num_populations"))[
            "num_populations"
        ]
        .max()
    )
    return max_val is not None and float(cast(float, max_val)) > 1


def _check_if_multiple_populations_are_being_validated(
    aj_estimates: pl.DataFrame,
) -> bool:
    return aj_estimates["aj_estimate"].unique().len() > 1


def _check_if_multiple_models_are_being_validated(aj_estimates: pl.DataFrame) -> bool:
    return aj_estimates["reference_group"].unique().len() > 1


def _infer_performance_data_type_times(
    aj_estimates_from_performance_data: pl.DataFrame, multiple_populations: bool
) -> str:
    multiple_models = _check_if_multiple_populations_are_being_validated_times(
        aj_estimates_from_performance_data
    )

    if multiple_populations:
        return "several populations"
    elif multiple_models:
        return "several models"
    else:
        return "single model"


def _infer_performance_data_type(
    aj_estimates_from_performance_data: pl.DataFrame, multiple_populations: bool
) -> str:
    multiple_models = _check_if_multiple_models_are_being_validated(
        aj_estimates_from_performance_data
    )

    if multiple_populations:
        return "several populations"
    elif multiple_models:
        return "several models"
    else:
        return "single model"


def _bold_hover_metrics(text: str, metrics: Sequence[str]) -> str:
    lines = text.split("<br>")
    for metric in metrics:
        label = _HOVER_LABELS.get(metric, metric)
        lines = [
            f"<b>{line}</b>" if label in line and "<b>" not in line else line
            for line in lines
        ]
    return "<br>".join(lines)


def _add_model_population_text(text: str, row: dict, perf_dat_type: str) -> str:
    if perf_dat_type == "several models" and "model" in row:
        text = f"<b>Model: {row['model']}</b><br>{text}"
    if perf_dat_type == "several populations" and "population" in row:
        text = f"<b>Population: {row['population']}</b><br>{text}"
    return text


def _round_val(value: Any, digits: int = 3):
    try:
        if value is None:
            return ""
        if isinstance(value, (int, float, np.floating)):
            return round(float(value), digits)
    except (TypeError, ValueError):
        pass
    return value


def _build_hover_text(
    row: dict,
    performance_metric_x: str,
    performance_metric_y: str,
    stratified_by: str,
    perf_dat_type: str,
) -> str:
    interventions_avoided = performance_metric_y == "net_benefit_interventions_avoided"

    raw_probability_threshold = row.get("chosen_cutoff")
    probability_threshold = _round_val(raw_probability_threshold)
    sensitivity = _round_val(row.get("sensitivity"))
    fpr = _round_val(row.get("false_positive_rate"))
    specificity = _round_val(row.get("specificity"))
    lift = _round_val(row.get("lift"))
    ppv = _round_val(row.get("ppv"))
    npv = _round_val(row.get("npv"))
    net_benefit = _round_val(row.get("net_benefit"))
    nb_interventions_avoided = _round_val(row.get("net_benefit_interventions_avoided"))
    predicted_positives = _round_val(row.get("predicted_positives"))
    raw_ppcr = row.get("ppcr")
    ppcr_percent = (
        _round_val(100 * raw_ppcr)
        if isinstance(raw_ppcr, (int, float, np.floating))
        else ""
    )
    tp = _round_val(row.get("true_positives"))
    tn = _round_val(row.get("true_negatives"))
    fp = _round_val(row.get("false_positives"))
    fn = _round_val(row.get("false_negatives"))

    if (
        isinstance(raw_probability_threshold, (int, float, np.floating))
        and raw_probability_threshold != 0
    ):
        odds = _round_val(
            (1 - raw_probability_threshold) / raw_probability_threshold, 2
        )
    else:
        odds = None

    if not interventions_avoided:
        text_lines = [
            f"Prob. Threshold: {probability_threshold}",
            f"Sensitivity: {sensitivity}",
            f"1 - Specificity (FPR): {fpr}",
            f"Specificity: {specificity}",
            f"Lift: {lift}",
            f"PPV: {ppv}",
            f"NPV: {npv}",
        ]
        if stratified_by == "probability_threshold":
            text_lines.append(f"NB: {net_benefit}")
            if odds is not None and math.isfinite(float(odds)):
                text_lines.append(f"Odds of Prob. Threshold: 1:{odds}")
        text_lines.extend(
            [
                f"Predicted Positives: {predicted_positives} ({ppcr_percent}%)",
                f"TP: {tp}",
                f"TN: {tn}",
                f"FP: {fp}",
                f"FN: {fn}",
            ]
        )
    else:
        text_lines = [
            f"Prob. Threshold: {probability_threshold}",
            f"Interventions Avoided (per 100): {nb_interventions_avoided}",
            f"NB: {net_benefit}",
            f"Predicted Positives: {predicted_positives} ({ppcr_percent}%)",
            f"TN: {tn}",
            f"FN: {fn}",
        ]
        if odds is not None and math.isfinite(float(odds)):
            text_lines.insert(1, f"Odds of Prob. Threshold: 1:{odds}")

    text = "<br>".join(text_lines)
    text = _bold_hover_metrics(text, [performance_metric_x, performance_metric_y])
    text = _add_model_population_text(text, row, perf_dat_type)
    return text.replace("NaN", "").replace("nan", "")


def _add_hover_text_to_performance_data(
    performance_data: pl.DataFrame,
    performance_metric_x: str,
    performance_metric_y: str,
    stratified_by: str,
    perf_dat_type: str,
) -> pl.DataFrame:
    hover_text_expr = pl.struct(performance_data.columns).map_elements(
        lambda row: _build_hover_text(
            row,
            performance_metric_x=performance_metric_x,
            performance_metric_y=performance_metric_y,
            stratified_by=stratified_by,
            perf_dat_type=perf_dat_type,
        ),
        return_dtype=pl.Utf8,
    )

    return performance_data.with_columns(
        [pl.col(pl.Float64).round(3), hover_text_expr.alias("text")]
    )


def _create_rtichoke_curve_list_times(
    performance_data: pl.DataFrame,
    stratified_by: str,
    size: int = 500,
    color_value=None,
    curve="roc",
    min_p_threshold=0,
    max_p_threshold=1,
) -> dict[str, Any]:
    animation_slider_cutoff_prefix = (
        "Prob. Threshold: "
        if stratified_by == "probability_threshold"
        else "Predicted Positives (Rate):"
    )

    x_metric, y_metric, x_label, y_label = _CURVE_CONFIG[curve]

    aj_estimates_from_performance_data = _get_aj_estimates_from_performance_data_times(
        performance_data
    )

    multiple_populations = _check_if_multiple_populations_are_being_validated_times(
        aj_estimates_from_performance_data
    )

    multiple_models = _check_if_multiple_models_are_being_validated(
        aj_estimates_from_performance_data
    )

    perf_dat_type = _infer_performance_data_type_times(
        aj_estimates_from_performance_data, multiple_populations
    )

    multiple_reference_groups = multiple_populations or multiple_models

    performance_data_with_hover_text = _add_hover_text_to_performance_data(
        performance_data.sort("chosen_cutoff"),
        performance_metric_x=x_metric,
        performance_metric_y=y_metric,
        stratified_by=stratified_by,
        perf_dat_type=perf_dat_type,
    )

    performance_data_ready_for_curve = _select_and_rename_necessary_variables(
        performance_data_with_hover_text, x_metric, y_metric
    )

    axes_ranges = extract_axes_ranges(
        performance_data_ready_for_curve,
        curve=curve,
        min_p_threshold=min_p_threshold,
        max_p_threshold=max_p_threshold,
    )

    reference_group_keys = performance_data["reference_group"].unique().to_list()

    fixed_time_horizons = (
        performance_data_ready_for_curve.select(pl.col("fixed_time_horizon"))
        .unique()
        .sort("fixed_time_horizon")
        .to_series()
        .to_list()
    )

    if "fixed_time_horizon" in performance_data_ready_for_curve.columns:
        reference_data_per_horizon = []

        for fixed_time_horizon in fixed_time_horizons:
            reference_lines = _create_reference_lines_data(
                curve=curve,
                aj_estimates_from_performance_data=(
                    aj_estimates_from_performance_data.filter(
                        pl.col("fixed_time_horizon") == fixed_time_horizon
                    )
                ),
                multiple_populations=multiple_populations,
                min_p_threshold=min_p_threshold,
                max_p_threshold=max_p_threshold,
            ).with_columns(pl.lit(fixed_time_horizon).alias("fixed_time_horizon"))

            reference_data_per_horizon.append(reference_lines)

        reference_data = (
            pl.concat(reference_data_per_horizon, how="vertical")
            if reference_data_per_horizon
            else pl.DataFrame()
        )

    else:
        reference_data = _create_reference_lines_data(
            curve=curve,
            aj_estimates_from_performance_data=aj_estimates_from_performance_data,
            multiple_populations=multiple_populations,
            min_p_threshold=min_p_threshold,
            max_p_threshold=max_p_threshold,
        )

    if (
        "fixed_time_horizon" in performance_data_ready_for_curve.columns
        and "fixed_time_horizon" not in reference_data.columns
    ):
        reference_data = reference_data.join(
            pl.DataFrame({"fixed_time_horizon": fixed_time_horizons}), how="cross"
        )

    cutoffs = (
        performance_data_ready_for_curve.select(pl.col("chosen_cutoff"))
        .drop_nulls()
        .unique()
        .sort("chosen_cutoff")
        .to_series()
        .to_list()
    )

    palette = [
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

    colors_dictionary = {
        **{
            key: "#BEBEBE"
            for key in [
                "random_guess",
                "perfect_model",
                "treat_none",
                "treat_all",
            ]
        },
        **{
            variant_key: (
                palette[group_index] if multiple_reference_groups else "#000000"
            )
            for group_index, reference_group in enumerate(reference_group_keys)
            for variant_key in [
                reference_group,
                f"random_guess_{reference_group}",
                f"perfect_model_{reference_group}",
                f"treat_none_{reference_group}",
                f"treat_all_{reference_group}",
            ]
        },
    }

    rtichoke_curve_list = {
        "size": size,
        "axes_ranges": axes_ranges,
        "x_label": x_label,
        "y_label": y_label,
        "animation_slider_cutoff_prefix": animation_slider_cutoff_prefix,
        "fixed_time_horizons": fixed_time_horizons,
        "reference_group_keys": reference_group_keys,
        "performance_data_ready_for_curve": performance_data_ready_for_curve,
        "reference_data": reference_data,
        "cutoffs": cutoffs,
        "colors_dictionary": colors_dictionary,
        "multiple_reference_groups": multiple_reference_groups,
    }

    return rtichoke_curve_list


def _create_rtichoke_curve_list_binary(
    performance_data: pl.DataFrame,
    stratified_by: str,
    size: int = 500,
    color_values=None,
    curve="roc",
    min_p_threshold=0,
    max_p_threshold=1,
) -> dict[str, Any]:
    animation_slider_prefix = (
        "Prob. Threshold: "
        if stratified_by == "probability_threshold"
        else "Predicted Positives (Rate):"
    )

    x_metric, y_metric, x_label, y_label = _CURVE_CONFIG[curve]

    aj_estimates_from_performance_data = _get_aj_estimates_from_performance_data(
        performance_data
    )

    multiple_populations = _check_if_multiple_populations_are_being_validated(
        aj_estimates_from_performance_data
    )

    multiple_models = _check_if_multiple_models_are_being_validated(
        aj_estimates_from_performance_data
    )

    perf_dat_type = _infer_performance_data_type(
        aj_estimates_from_performance_data, multiple_populations
    )

    multiple_reference_groups = multiple_populations or multiple_models

    performance_data_with_hover_text = _add_hover_text_to_performance_data(
        performance_data.sort("chosen_cutoff"),
        performance_metric_x=x_metric,
        performance_metric_y=y_metric,
        stratified_by=stratified_by,
        perf_dat_type=perf_dat_type,
    )

    performance_data_ready_for_curve = _select_and_rename_necessary_variables(
        performance_data_with_hover_text, x_metric, y_metric
    )

    reference_data = _create_reference_lines_data(
        curve=curve,
        aj_estimates_from_performance_data=aj_estimates_from_performance_data,
        multiple_populations=multiple_populations,
        min_p_threshold=min_p_threshold,
        max_p_threshold=max_p_threshold,
    )

    axes_ranges = extract_axes_ranges(
        performance_data_ready_for_curve,
        curve=curve,
        min_p_threshold=min_p_threshold,
        max_p_threshold=max_p_threshold,
    )

    reference_group_keys = performance_data["reference_group"].unique().to_list()

    cutoffs = (
        performance_data_ready_for_curve.select(pl.col("chosen_cutoff"))
        .drop_nulls()
        .unique()
        .sort("chosen_cutoff")
        .to_series()
        .to_list()
    )

    palette = [
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

    colors_dictionary = {
        **{
            key: "#BEBEBE"
            for key in [
                "random_guess",
                "perfect_model",
                "treat_none",
                "treat_all",
            ]
        },
        **{
            variant_key: (
                palette[group_index] if multiple_reference_groups else "#000000"
            )
            for group_index, reference_group in enumerate(reference_group_keys)
            for variant_key in [
                reference_group,
                f"random_guess_{reference_group}",
                f"perfect_model_{reference_group}",
                f"treat_none_{reference_group}",
                f"treat_all_{reference_group}",
            ]
        },
    }

    rtichoke_curve_list = {
        "size": size,
        "axes_ranges": axes_ranges,
        "x_label": x_label,
        "y_label": y_label,
        "animation_slider_prefix": animation_slider_prefix,
        "reference_group_keys": reference_group_keys,
        "performance_data_ready_for_curve": performance_data_ready_for_curve,
        "reference_data": reference_data,
        "cutoffs": cutoffs,
        "colors_dictionary": colors_dictionary,
        "multiple_reference_groups": multiple_reference_groups,
    }

    return rtichoke_curve_list


def _select_and_rename_necessary_variables(
    performance_data: pl.DataFrame, x_perf_metric: str, y_perf_metric: str
) -> pl.DataFrame:
    selected_columns = [
        pl.col("reference_group"),
        pl.col("chosen_cutoff"),
        pl.col(x_perf_metric).alias("x"),
        pl.col(y_perf_metric).alias("y"),
        pl.col("text"),
    ]

    if "fixed_time_horizon" in performance_data.columns:
        selected_columns.append(pl.col("fixed_time_horizon"))

    return performance_data.select(*selected_columns)


def _create_slider_dict(
    animation_slider_prefic: str, steps: list[dict[str, Any]]
) -> dict[str, Any]:
    slider_dict = {
        "active": 0,
        "yanchor": "top",
        "xanchor": "left",
        "currentvalue": {
            "font": {"size": 20},
            "prefix": animation_slider_prefic,
            "visible": True,
            "xanchor": "left",
        },
        "transition": {"duration": 300, "easing": "linear"},
        "pad": {"b": 10, "t": 50},
        "len": 0.9,
        "x": 0.1,
        "y": 0,
        "steps": steps,
    }

    return slider_dict


def _create_plotly_curve_times(rtichoke_curve_list: dict[str, Any]) -> go.Figure:
    initial_fixed_time_horizon = rtichoke_curve_list["fixed_time_horizons"][0]
    initial_cutoff = (
        rtichoke_curve_list["cutoffs"][0] if rtichoke_curve_list["cutoffs"] else None
    )

    def _xy_for_curve(
        group: str, fixed_time_horizon: float
    ) -> tuple[list[Any], list[Any]]:
        subset = rtichoke_curve_list["performance_data_ready_for_curve"].filter(
            (pl.col("reference_group") == group)
            & (pl.col("fixed_time_horizon") == fixed_time_horizon)
        )
        return subset["x"].to_list(), subset["y"].to_list()

    def _xy_for_reference(
        group: str, fixed_time_horizon: float
    ) -> tuple[list[Any], list[Any], list[Any]]:
        subset = rtichoke_curve_list["reference_data"].filter(
            (pl.col("reference_group") == group)
            & (pl.col("fixed_time_horizon") == fixed_time_horizon)
        )
        return subset["x"].to_list(), subset["y"].to_list(), subset["text"].to_list()

    def _xy_at_cutoff(
        group: str, cutoff: float, fixed_time_horizon: float
    ) -> tuple[Any, Any, Any]:
        row = (
            rtichoke_curve_list["performance_data_ready_for_curve"]
            .filter(
                (pl.col("reference_group") == group)
                & (pl.col("fixed_time_horizon") == fixed_time_horizon)
                & (pl.col("chosen_cutoff") == cutoff)
                & pl.col("x").is_not_null()
                & pl.col("y").is_not_null()
            )
            .select(["x", "y", "text"])
            .limit(1)
        )
        if row.height == 0:
            return None, None, None
        r = row.row(0)
        return r[0], r[1], r[2]

    non_interactive_curve = []
    for fixed_time_horizon in rtichoke_curve_list["fixed_time_horizons"]:
        for group in rtichoke_curve_list["reference_group_keys"]:
            non_interactive_curve.append(
                go.Scatter(
                    x=_xy_for_curve(group, fixed_time_horizon)[0],
                    y=_xy_for_curve(group, fixed_time_horizon)[1],
                    text=rtichoke_curve_list["performance_data_ready_for_curve"].filter(
                        (pl.col("reference_group") == group)
                        & (pl.col("fixed_time_horizon") == fixed_time_horizon)
                    )["text"],
                    mode="markers+lines",
                    name=group,
                    legendgroup=group,
                    line={
                        "width": 2,
                        "color": rtichoke_curve_list["colors_dictionary"].get(group),
                    },
                    hoverlabel=dict(
                        bgcolor=rtichoke_curve_list["colors_dictionary"].get(group),
                        bordercolor=rtichoke_curve_list["colors_dictionary"].get(group),
                        font_color="white",
                    ),
                    hoverinfo="text",
                    showlegend=rtichoke_curve_list["multiple_reference_groups"],
                    visible=fixed_time_horizon == initial_fixed_time_horizon,
                )
            )

    marker_traces: list[go.Scatter] = []
    for fixed_time_horizon in rtichoke_curve_list["fixed_time_horizons"]:
        for group in rtichoke_curve_list["reference_group_keys"]:
            x_val, y_val, text_val = (
                _xy_at_cutoff(group, initial_cutoff, fixed_time_horizon)
                if initial_cutoff is not None
                else (None, None, None)
            )
            marker_traces.append(
                go.Scatter(
                    x=[x_val] if x_val is not None else [],
                    y=[y_val] if y_val is not None else [],
                    mode="markers",
                    marker={
                        "size": 12,
                        "color": (
                            rtichoke_curve_list["colors_dictionary"].get(group)
                            if rtichoke_curve_list["multiple_reference_groups"]
                            else "#f6e3be"
                        ),
                        "line": {"width": 3, "color": "black"},
                    },
                    name=f"{group} @ cutoff",
                    legendgroup=group,
                    hoverlabel=dict(
                        bgcolor="#f6e3be"
                        if not rtichoke_curve_list["multiple_reference_groups"]
                        else rtichoke_curve_list["colors_dictionary"].get(group),
                        bordercolor="#f6e3be"
                        if not rtichoke_curve_list["multiple_reference_groups"]
                        else rtichoke_curve_list["colors_dictionary"].get(group),
                        font_color="black"
                        if not rtichoke_curve_list["multiple_reference_groups"]
                        else "white",
                    ),
                    showlegend=False,
                    hoverinfo="text",
                    text=text_val,
                    visible=fixed_time_horizon == initial_fixed_time_horizon,
                )
            )

    reference_traces = []
    for fixed_time_horizon in rtichoke_curve_list["fixed_time_horizons"]:
        for group in rtichoke_curve_list["colors_dictionary"].keys():
            reference_traces.append(
                go.Scatter(
                    x=_xy_for_reference(group, fixed_time_horizon)[0],
                    y=_xy_for_reference(group, fixed_time_horizon)[1],
                    mode="lines",
                    name=group,
                    legendgroup=group,
                    line=dict(
                        dash="dot",
                        color=rtichoke_curve_list["colors_dictionary"].get(group),
                        width=1.5,
                    ),
                    hoverlabel=dict(
                        bgcolor=rtichoke_curve_list["colors_dictionary"].get(group),
                        bordercolor=rtichoke_curve_list["colors_dictionary"].get(group),
                        font_color="white",
                    ),
                    hoverinfo="text",
                    text=_xy_for_reference(group, fixed_time_horizon)[2],
                    showlegend=False,
                    visible=fixed_time_horizon == initial_fixed_time_horizon,
                )
            )

    num_curve_traces = len(non_interactive_curve)
    num_marker_traces = len(marker_traces)
    cutoff_target_indices = list(
        range(
            num_curve_traces,
            num_curve_traces + num_marker_traces,
        )
    )

    def marker_values_for_cutoff(
        cutoff: float,
    ) -> tuple[list[list], list[list], list[list]]:
        marker_values = [
            _xy_at_cutoff(group, cutoff, fixed_time_horizon)
            if cutoff is not None
            else (None, None, None)
            for fixed_time_horizon in rtichoke_curve_list["fixed_time_horizons"]
            for group in rtichoke_curve_list["reference_group_keys"]
        ]

        xs = [[x] if x is not None else [] for x, _, _ in marker_values]
        ys = [[y] if y is not None else [] for _, y, _ in marker_values]
        texts = [[text] if text is not None else [] for _, _, text in marker_values]

        return xs, ys, texts

    cutoff_steps = []
    for cutoff in rtichoke_curve_list["cutoffs"]:
        xs, ys, texts = marker_values_for_cutoff(cutoff)
        cutoff_steps.append(
            {
                "method": "restyle",
                "args": [
                    {
                        "x": xs,
                        "y": ys,
                        "text": texts,
                    },
                    cutoff_target_indices,
                ],
                "label": f"{cutoff:g}",
            }
        )

    steps_fixed_time_horizon = []
    total_traces = num_curve_traces + num_marker_traces + len(reference_traces)
    for fixed_time_horizon in rtichoke_curve_list["fixed_time_horizons"]:
        visibility: list[bool] = []
        for horizon in rtichoke_curve_list["fixed_time_horizons"]:
            horizon_visible = horizon == fixed_time_horizon
            visibility.extend(
                [horizon_visible] * len(rtichoke_curve_list["reference_group_keys"])
            )
        for horizon in rtichoke_curve_list["fixed_time_horizons"]:
            horizon_visible = horizon == fixed_time_horizon
            visibility.extend(
                [horizon_visible] * len(rtichoke_curve_list["reference_group_keys"])
            )
        for horizon in rtichoke_curve_list["fixed_time_horizons"]:
            horizon_visible = horizon == fixed_time_horizon
            visibility.extend(
                [horizon_visible] * len(rtichoke_curve_list["colors_dictionary"].keys())
            )

        steps_fixed_time_horizon.append(
            {
                "method": "restyle",
                "args": [
                    {"visible": visibility},
                    list(range(total_traces)),
                ],
                "label": f"{fixed_time_horizon:g}",
            }
        )

    slider_cutoff_dict = _create_slider_dict(
        rtichoke_curve_list["animation_slider_cutoff_prefix"], cutoff_steps
    )
    slider_fixed_time_horizon_dict = _create_slider_dict(
        "Fixed Time Horizon: ", steps_fixed_time_horizon
    )

    curve_layout = _create_curve_layout(
        size=rtichoke_curve_list["size"],
        slider_dict=[slider_cutoff_dict, slider_fixed_time_horizon_dict],
        axes_ranges=rtichoke_curve_list["axes_ranges"],
        x_label=rtichoke_curve_list["x_label"],
        y_label=rtichoke_curve_list["y_label"],
        show_legend=rtichoke_curve_list["multiple_reference_groups"],
    )

    return go.Figure(
        data=non_interactive_curve + marker_traces + reference_traces,
        layout=curve_layout,
    )


def _create_plotly_curve_binary(rtichoke_curve_list: dict[str, Any]) -> go.Figure:
    initial_cutoff = (
        rtichoke_curve_list["cutoffs"][0] if rtichoke_curve_list["cutoffs"] else None
    )

    non_interactive_curve = [
        go.Scatter(
            x=rtichoke_curve_list["performance_data_ready_for_curve"]
            .filter(pl.col("reference_group") == group)["x"]
            .to_list(),
            y=rtichoke_curve_list["performance_data_ready_for_curve"]
            .filter(pl.col("reference_group") == group)["y"]
            .to_list(),
            text=rtichoke_curve_list["performance_data_ready_for_curve"].filter(
                pl.col("reference_group") == group
            )["text"],
            mode="markers+lines",
            name=group,
            legendgroup=group,
            line={
                "width": 2,
                "color": rtichoke_curve_list["colors_dictionary"].get(group),
            },
            hoverlabel=dict(
                bgcolor=rtichoke_curve_list["colors_dictionary"].get(group),
                bordercolor=rtichoke_curve_list["colors_dictionary"].get(group),
                font_color="white",
            ),
            hoverinfo="text",
            showlegend=rtichoke_curve_list["multiple_reference_groups"],
        )
        for group in rtichoke_curve_list["reference_group_keys"]
    ]

    def xy_at_cutoff(group, c):
        row = (
            rtichoke_curve_list["performance_data_ready_for_curve"]
            .filter(
                (pl.col("reference_group") == group)
                & (pl.col("chosen_cutoff") == c)
                & pl.col("x").is_not_null()
                & pl.col("y").is_not_null()
            )
            .select(["x", "y", "text"])
            .limit(1)
        )
        if row.height == 0:
            return None, None, None
        r = row.row(0)
        return r[0], r[1], r[2]

    def marker_values_for_cutoff(
        cutoff: float,
    ) -> tuple[list[list], list[list], list[list]]:
        marker_values = [
            xy_at_cutoff(group, cutoff)
            for group in rtichoke_curve_list["reference_group_keys"]
        ]

        xs = [[x] if x is not None else [] for x, _, _ in marker_values]
        ys = [[y] if y is not None else [] for _, y, _ in marker_values]
        texts = [[text] if text is not None else [] for _, _, text in marker_values]

        return xs, ys, texts

    initial_xs, initial_ys, initial_texts = (
        marker_values_for_cutoff(initial_cutoff)
        if initial_cutoff is not None
        else (
            [[] for _ in rtichoke_curve_list["reference_group_keys"]],
            [[] for _ in rtichoke_curve_list["reference_group_keys"]],
            [[] for _ in rtichoke_curve_list["reference_group_keys"]],
        )
    )

    initial_interactive_markers = [
        go.Scatter(
            x=initial_xs[idx],
            y=initial_ys[idx],
            text=initial_texts[idx],
            # hovertext=initial_texts[idx],
            mode="markers",
            marker={
                "size": 12,
                "color": (
                    rtichoke_curve_list["colors_dictionary"].get(group)
                    if rtichoke_curve_list["multiple_reference_groups"]
                    else "#f6e3be"
                ),
                "line": {"width": 3, "color": "black"},
            },
            name=f"{group} @ cutoff",
            legendgroup=group,
            hoverlabel=dict(
                bgcolor="#f6e3be"
                if not rtichoke_curve_list["multiple_reference_groups"]
                else rtichoke_curve_list["colors_dictionary"].get(group),
                bordercolor="#f6e3be"
                if not rtichoke_curve_list["multiple_reference_groups"]
                else rtichoke_curve_list["colors_dictionary"].get(group),
                font_color="black"
                if not rtichoke_curve_list["multiple_reference_groups"]
                else "white",
            ),
            showlegend=False,
            hoverinfo="text",
        )
        for idx, group in enumerate(rtichoke_curve_list["reference_group_keys"])
    ]

    reference_traces = [
        go.Scatter(
            x=rtichoke_curve_list["reference_data"]
            .filter(pl.col("reference_group") == group)["x"]
            .to_list(),
            y=rtichoke_curve_list["reference_data"]
            .filter(pl.col("reference_group") == group)["y"]
            .to_list(),
            mode="lines",
            name=group,
            legendgroup=group,
            line=dict(
                dash="dot",
                color=rtichoke_curve_list["colors_dictionary"].get(group),
                width=1.5,
            ),
            hoverlabel=dict(
                bgcolor=rtichoke_curve_list["colors_dictionary"].get(group),
                bordercolor=rtichoke_curve_list["colors_dictionary"].get(group),
                font_color="white",
            ),
            hoverinfo="text",
            text=rtichoke_curve_list["reference_data"]
            .filter(pl.col("reference_group") == group)["text"]
            .to_list(),
            showlegend=False,
        )
        for group in rtichoke_curve_list["colors_dictionary"].keys()
    ]

    dyn_idx = list(
        range(
            len(rtichoke_curve_list["reference_group_keys"]),
            len(rtichoke_curve_list["reference_group_keys"]) * 2,
        )
    )

    steps = []
    for cutoff in rtichoke_curve_list["cutoffs"]:
        xs, ys, texts = marker_values_for_cutoff(cutoff)
        steps.append(
            {
                "method": "restyle",
                "args": [
                    {
                        "x": xs,
                        "y": ys,
                        "text": texts,
                        # "hovertext": texts,
                    },
                    dyn_idx,
                ],
                "label": f"{cutoff:g}",
            }
        )

    slider_dict = _create_slider_dict(
        rtichoke_curve_list["animation_slider_prefix"], steps
    )

    curve_layout = _create_curve_layout(
        size=rtichoke_curve_list["size"],
        slider_dict=slider_dict,
        axes_ranges=rtichoke_curve_list["axes_ranges"],
        x_label=rtichoke_curve_list["x_label"],
        y_label=rtichoke_curve_list["y_label"],
        show_legend=rtichoke_curve_list["multiple_reference_groups"],
    )

    return go.Figure(
        data=non_interactive_curve + initial_interactive_markers + reference_traces,
        layout=curve_layout,
    )


def _create_curve_layout(
    size: int,
    slider_dict: dict | list[dict],
    axes_ranges: dict[str, list[float]] | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    show_legend: bool = True,
) -> dict[str, Any]:
    sliders = slider_dict if isinstance(slider_dict, list) else [slider_dict]

    if len(sliders) > 1:
        vertical_spacing = -0.4
        for idx, slider in enumerate(sliders):
            slider_y = slider.get("y", 0)
            base_y = slider_y if isinstance(slider_y, (int, float)) else -0.2
            slider["y"] = base_y + vertical_spacing * float(idx)
            base_pad = (
                slider.get("pad", {}) if isinstance(slider.get("pad"), dict) else {}
            )
            slider["pad"] = {
                "t": max(120, base_pad.get("t", 0)),
                "b": max(80, base_pad.get("b", 0)),
                **base_pad,
            }
    xaxis: dict[str, Any] = {"showgrid": False}
    yaxis: dict[str, Any] = {"showgrid": False}

    if axes_ranges is not None:
        xaxis["range"] = axes_ranges["xaxis"]
        yaxis["range"] = axes_ranges["yaxis"]

    if x_label:
        xaxis["title"] = {"text": x_label}
    if y_label:
        yaxis["title"] = {"text": y_label}

    curve_layout = {
        "xaxis": xaxis,
        "yaxis": yaxis,
        "template": "plotly",
        "plot_bgcolor": "rgba(0, 0, 0, 0)",
        "paper_bgcolor": "rgba(0, 0, 0, 0)",
        "showlegend": True,
        "legend": {
            "orientation": "h",
            "xanchor": "center",
            "yanchor": "top",
            "x": 0.5,
            "y": 1.3,
            "bgcolor": "rgba(0, 0, 0, 0)",
            "bordercolor": "rgba(0, 0, 0, 0)",
        },
        "height": size + 100,
        "width": size,
        # "hoverlabel": {"bgcolor": "rgba(0,0,0,0)", "bordercolor": "rgba(0,0,0,0)"},
        "updatemenus": [
            {
                "type": "buttons",
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "visible": False,
                        "args": [None, {"frame": {"duration": 500, "redraw": False}}],
                    }
                ],
            }
        ],
        "sliders": sliders,
        "modebar": {"remove": list(DEFAULT_MODEBAR_BUTTONS_TO_REMOVE)},
    }

    return curve_layout


def _create_interactive_marker_polars(
    performance_data_ready_for_curve: pl.DataFrame,
    interactive_marker_color: str,
    k: int,
    reference_group: str,
):
    interactive_marker = go.Scatter(
        x=[performance_data_ready_for_curve["x"][k]],
        y=[performance_data_ready_for_curve["y"][k]],
        mode="markers",
        # hoverinfo="text",
        # hovertext=[performance_data_ready_for_curve["text"].values.tolist()[k]],
        name=reference_group,
        legendgroup=reference_group,
        showlegend=False,
        marker={
            "size": 12,
            "color": interactive_marker_color,
            "line": {"width": 2, "color": "black"},
        },
    )
    return interactive_marker

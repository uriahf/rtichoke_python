"""
A module for helpers related to plotly
"""

import plotly.graph_objects as go
import polars as pl
import math
from typing import Any, Dict, Union, Sequence
import numpy as np
from rtichoke.performance_data.performance_data import prepare_performance_data


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


def _plot_rtichoke_curve_binary(
    performance_data: pl.DataFrame,
    stratified_by: Sequence[str] = ["probability_threshold"],
    curve: str = "roc",
    size: int = 600,
) -> go.Figure:
    rtichoke_curve_list = _create_rtichoke_curve_list_binary(
        performance_data=performance_data,
        stratified_by=stratified_by[0],
        curve=curve,
        size=size,
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
                and float(aj_df["p"].max()) == 0.0
                and float(aj_df["p"].min()) == 0.0
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


def _check_if_multiple_populations_are_being_validated(
    aj_estimates: pl.DataFrame,
) -> bool:
    return aj_estimates["aj_estimate"].unique().len() > 1


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

    performance_data_ready_for_curve = _select_and_rename_necessary_variables(
        performance_data.sort("chosen_cutoff"), x_metric, y_metric
    )

    aj_estimates_from_performance_data = _get_aj_estimates_from_performance_data(
        performance_data
    )

    multiple_populations = _check_if_multiple_populations_are_being_validated(
        aj_estimates_from_performance_data
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
            variant_key: (palette[group_index] if multiple_populations else "#000000")
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
        "multiple_populations": multiple_populations,
    }

    return rtichoke_curve_list


def _select_and_rename_necessary_variables(
    performance_data: pl.DataFrame, x_perf_metric: str, y_perf_metric: str
) -> pl.DataFrame:
    return performance_data.select(
        pl.col("reference_group"),
        pl.col("chosen_cutoff"),
        pl.col(x_perf_metric).alias("x"),
        pl.col(y_perf_metric).alias("y"),
    )


def _create_slider_dict(animation_slider_prefic: str, steps: dict) -> dict[str, Any]:
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


def _create_plotly_curve_binary(rtichoke_curve_list: dict[str, Any]) -> go.Figure:
    non_interactive_curve = [
        go.Scatter(
            x=rtichoke_curve_list["performance_data_ready_for_curve"]
            .filter(pl.col("reference_group") == group)["x"]
            .to_list(),
            y=rtichoke_curve_list["performance_data_ready_for_curve"]
            .filter(pl.col("reference_group") == group)["y"]
            .to_list(),
            mode="markers+lines",
            name=group,
            legendgroup=group,
            line={
                "width": 2,
                "color": rtichoke_curve_list["colors_dictionary"].get(group),
            },
            showlegend=True,
        )
        for group in rtichoke_curve_list["reference_group_keys"]
    ]

    initial_interactive_markers = [
        go.Scatter(
            x=[],
            y=[],
            mode="markers",
            marker={
                "size": 12,
                "color": (
                    rtichoke_curve_list["colors_dictionary"].get(group)
                    if rtichoke_curve_list["multiple_populations"]
                    else "#f6e3be"
                ),
                "line": {"width": 3, "color": "black"},
            },
            name=f"{group} @ cutoff",
            legendgroup=group,
            showlegend=False,
            hovertemplate=f"{group}<br>x=%{{x:.4f}}<br>y=%{{y:.4f}}<extra></extra>",
        )
        for group in rtichoke_curve_list["reference_group_keys"]
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

    def xy_at_cutoff(group, c):
        row = (
            rtichoke_curve_list["performance_data_ready_for_curve"]
            .filter(
                (pl.col("reference_group") == group)
                & (pl.col("chosen_cutoff") == c)
                & pl.col("x").is_not_null()
                & pl.col("y").is_not_null()
            )
            .select(["x", "y"])
            .limit(1)
        )
        if row.height == 0:
            return None, None
        r = row.row(0)  # (x, y)
        return r[0], r[1]

    steps = [
        {
            "method": "restyle",
            "args": [
                {
                    "x": [
                        [xy_at_cutoff(group, cutoff)[0]]
                        if xy_at_cutoff(group, cutoff)[0] is not None
                        else []
                        for group in rtichoke_curve_list["reference_group_keys"]
                    ],
                    "y": [
                        [xy_at_cutoff(group, cutoff)[1]]
                        if xy_at_cutoff(group, cutoff)[1] is not None
                        else []
                        for group in rtichoke_curve_list["reference_group_keys"]
                    ],
                },
                dyn_idx,
            ],
            "label": f"{cutoff:g}",
        }
        for cutoff in rtichoke_curve_list["cutoffs"]
    ]

    slider_dict = _create_slider_dict(
        rtichoke_curve_list["animation_slider_prefix"], steps
    )

    curve_layout = _create_curve_layout(
        size=rtichoke_curve_list["size"],
        slider_dict=slider_dict,
        axes_ranges=rtichoke_curve_list["axes_ranges"],
        x_label=rtichoke_curve_list["x_label"],
        y_label=rtichoke_curve_list["y_label"],
    )

    return go.Figure(
        data=non_interactive_curve + initial_interactive_markers + reference_traces,
        layout=curve_layout,
    )


def _create_curve_layout(
    size: int,
    slider_dict: dict,
    axes_ranges: dict[str, list[float]] | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
) -> dict[str, Any]:
    curve_layout = {
        "xaxis": {"showgrid": False},
        "yaxis": {"showgrid": False},
        "template": "none",
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
        "height": size + 50,
        "width": size,
        "hoverlabel": {"bgcolor": "rgba(0,0,0,0)", "bordercolor": "rgba(0,0,0,0)"},
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
        "sliders": [slider_dict],
    }

    if axes_ranges is not None:
        curve_layout["xaxis"]["range"] = axes_ranges["xaxis"]
        curve_layout["yaxis"]["range"] = axes_ranges["yaxis"]

    if x_label:
        curve_layout["xaxis"]["title"] = {"text": x_label}
    if y_label:
        curve_layout["yaxis"]["title"] = {"text": y_label}

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

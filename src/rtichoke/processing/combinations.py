import numpy as np
import polars as pl
from typing import Dict
from collections.abc import Sequence


def _enum_dataframe(column_name: str, values: Sequence[str]) -> pl.DataFrame:
    """Create a single-column DataFrame with an enum dtype."""
    enum_values = list(dict.fromkeys(values))
    enum_dtype = pl.Enum(enum_values)
    return pl.DataFrame({column_name: pl.Series(values, dtype=enum_dtype)})


def create_strata_combinations(stratified_by: str, by: float, breaks) -> pl.DataFrame:
    s_by = str(by)
    decimals = len(s_by.split(".")[-1]) if "." in s_by else 0
    fmt = f"{{:.{decimals}f}}"

    if stratified_by == "probability_threshold":
        upper_bound = breaks[1:]  # breaks
        lower_bound = breaks[:-1]  # np.roll(upper_bound, 1)
        # lower_bound[0] = 0.0
        mid_point = upper_bound - by / 2
        include_lower_bound = lower_bound > -0.1
        include_upper_bound = upper_bound == 1.0  # upper_bound != 0.0
        # chosen_cutoff = upper_bound
        strata = format_strata_column(
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            include_lower_bound=include_lower_bound,
            include_upper_bound=include_upper_bound,
            decimals=2,
        )

    elif stratified_by == "ppcr":
        strata_mid = breaks[1:]
        lower_bound = strata_mid - by / 2
        upper_bound = strata_mid + by / 2
        mid_point = breaks[1:]
        include_lower_bound = np.ones_like(strata_mid, dtype=bool)
        include_upper_bound = np.zeros_like(strata_mid, dtype=bool)
        # chosen_cutoff = strata_mid
        strata = np.array([fmt.format(x) for x in strata_mid], dtype=object)
    else:
        raise ValueError(f"Unsupported stratified_by: {stratified_by}")

    bins_df = pl.DataFrame(
        {
            "strata": pl.Series(strata),
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "mid_point": mid_point,
            "include_lower_bound": include_lower_bound,
            "include_upper_bound": include_upper_bound,
            # "chosen_cutoff": chosen_cutoff,
            "stratified_by": [stratified_by] * len(strata),
        }
    )

    cutoffs_df = pl.DataFrame({"chosen_cutoff": breaks})

    return bins_df.join(cutoffs_df, how="cross")


def format_strata_column(
    lower_bound: list[float],
    upper_bound: list[float],
    include_lower_bound: list[bool],
    include_upper_bound: list[bool],
    decimals: int = 3,
) -> list[str]:
    return [
        f"{'[' if ilb else '('}"
        f"{round(lb, decimals):.{decimals}f}, "
        f"{round(ub, decimals):.{decimals}f}"
        f"{']' if iub else ')'}"
        for lb, ub, ilb, iub in zip(
            lower_bound, upper_bound, include_lower_bound, include_upper_bound
        )
    ]


def format_strata_interval(
    lower: float, upper: float, include_lower: bool, include_upper: bool
) -> str:
    left = "[" if include_lower else "("
    right = "]" if include_upper else ")"
    return f"{left}{lower:.3f}, {upper:.3f}{right}"


def create_breaks_values(probs_vec, stratified_by, by):
    if stratified_by != "probability_threshold":
        breaks = np.quantile(probs_vec, np.linspace(1, 0, int(1 / by) + 1))
    else:
        breaks = np.round(
            np.arange(0, 1 + by, by), decimals=len(str(by).split(".")[-1])
        )
    return breaks


def _create_aj_data_combinations_binary(
    reference_groups: Sequence[str],
    stratified_by: Sequence[str],
    by: float,
    breaks: Sequence[float],
) -> pl.DataFrame:
    dfs = [create_strata_combinations(sb, by, breaks) for sb in stratified_by]

    strata_combinations = pl.concat(dfs, how="vertical")

    strata_cats = (
        strata_combinations.select(pl.col("strata").unique(maintain_order=True))
        .to_series()
        .to_list()
    )

    strata_enum = pl.Enum(strata_cats)
    stratified_by_enum = pl.Enum(["probability_threshold", "ppcr"])

    strata_combinations = strata_combinations.with_columns(
        [
            pl.col("strata").cast(strata_enum),
            pl.col("stratified_by").cast(stratified_by_enum),
        ]
    )

    # Define values for Cartesian product
    reals_labels = ["real_negatives", "real_positives"]

    combinations_frames: list[pl.DataFrame] = [
        _enum_dataframe("reference_group", reference_groups),
        strata_combinations,
        _enum_dataframe("reals_labels", reals_labels),
    ]

    result = combinations_frames[0]
    for frame in combinations_frames[1:]:
        result = result.join(frame, how="cross")

    return result


def create_aj_data_combinations(
    reference_groups: Sequence[str],
    heuristics_sets: list[Dict],
    fixed_time_horizons: Sequence[float],
    stratified_by: Sequence[str],
    by: float,
    breaks: Sequence[float],
    risk_set_scope: Sequence[str] = ["within_stratum", "pooled_by_cutoff"],
) -> pl.DataFrame:
    dfs = [create_strata_combinations(sb, by, breaks) for sb in stratified_by]
    strata_combinations = pl.concat(dfs, how="vertical")

    # strata_enum = pl.Enum(strata_combinations["strata"])

    strata_cats = (
        strata_combinations.select(pl.col("strata").unique(maintain_order=True))
        .to_series()
        .to_list()
    )

    strata_enum = pl.Enum(strata_cats)
    stratified_by_enum = pl.Enum(["probability_threshold", "ppcr"])

    strata_combinations = strata_combinations.with_columns(
        [
            pl.col("strata").cast(strata_enum),
            pl.col("stratified_by").cast(stratified_by_enum),
        ]
    )

    risk_set_scope_combinations = pl.DataFrame(
        {
            "risk_set_scope": pl.Series(risk_set_scope).cast(
                pl.Enum(["within_stratum", "pooled_by_cutoff"])
            )
        }
    )

    # Define values for Cartesian product
    reals_labels = [
        "real_negatives",
        "real_positives",
        "real_competing",
        "real_censored",
    ]

    heuristics_combinations = pl.DataFrame(heuristics_sets)

    censoring_heuristics_enum = pl.Enum(
        heuristics_combinations["censoring_heuristic"].unique(maintain_order=True)
    )
    competing_heuristics_enum = pl.Enum(
        heuristics_combinations["competing_heuristic"].unique(maintain_order=True)
    )

    combinations_frames: list[pl.DataFrame] = [
        _enum_dataframe("reference_group", reference_groups),
        pl.DataFrame(
            {"fixed_time_horizon": pl.Series(fixed_time_horizons, dtype=pl.Float64)}
        ),
        heuristics_combinations.with_columns(
            [
                pl.col("censoring_heuristic").cast(censoring_heuristics_enum),
                pl.col("competing_heuristic").cast(competing_heuristics_enum),
            ]
        ),
        strata_combinations,
        risk_set_scope_combinations,
        _enum_dataframe("reals_labels", reals_labels),
    ]

    result = combinations_frames[0]
    for frame in combinations_frames[1:]:
        result = result.join(frame, how="cross")

    return result

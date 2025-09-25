from lifelines import AalenJohansenFitter
import pandas as pd
import numpy as np
import polars as pl
from polarstate import predict_aj_estimates
from polarstate import prepare_event_table
from typing import Dict, Union
from collections.abc import Sequence


def _enum_dataframe(column_name: str, values: Sequence[str]) -> pl.DataFrame:
    """Create a single-column DataFrame with an enum dtype."""
    enum_values = list(dict.fromkeys(values))
    enum_dtype = pl.Enum(enum_values)
    return pl.DataFrame({column_name: pl.Series(values, dtype=enum_dtype)})


def extract_aj_estimate(data_to_adjust, fixed_time_horizons):
    """
    Python implementation of the R extract_aj_estimate function for Aalen-Johansen estimation.

    Parameters:
    data_to_adjust (pd.DataFrame): DataFrame containing survival data
    fixed_time_horizons (list or float): Time points at which to evaluate the survival

    Returns:
    pd.DataFrame: DataFrame with Aalen-Johansen estimates
    """

    # Ensure fixed_time_horizons is a list
    if not isinstance(fixed_time_horizons, list):
        fixed_time_horizons = [fixed_time_horizons]

    # Create a categorical version of reals for stratification
    data = data_to_adjust.copy()
    data["reals_cat"] = pd.Categorical(
        data["reals_labels"],
        categories=[
            "real_negatives",
            "real_positives",
            "real_competing",
            "real_censored",
        ],
        ordered=True,
    )

    # Get unique strata values
    strata_values = data["strata"].unique()

    event_map = {
        "real_negatives": 0,  # Treat as censored
        "real_positives": 1,  # Event of interest
        "real_competing": 2,  # Competing risk
        "real_censored": 0,  # Censored
    }

    data["event_code"] = data["reals_labels"].map(event_map)

    # Initialize result dataframes
    results = []

    # For each stratum, fit Aalen-Johansen model
    for stratum in strata_values:
        # Filter data for current stratum
        stratum_data = data.loc[data["strata"] == stratum]

        # Initialize Aalen-Johansen fitter
        ajf = AalenJohansenFitter()
        ajf_competing = AalenJohansenFitter()

        # Fit the model
        ajf.fit(stratum_data["times"], stratum_data["event_code"], event_of_interest=1)

        ajf_competing.fit(
            stratum_data["times"], stratum_data["event_code"], event_of_interest=2
        )

        # Calculate cumulative incidence at fixed time horizons
        for t in fixed_time_horizons:
            n = len(stratum_data)
            real_positives_est = ajf.predict(t)
            real_competing_est = ajf_competing.predict(t)
            real_negatives_est = 1 - real_positives_est - real_competing_est

            states = ["real_negatives", "real_positives", "real_competing"]
            estimates = [real_negatives_est, real_positives_est, real_competing_est]

            for state, estimate in zip(states, estimates):
                results.append(
                    {
                        "strata": stratum,
                        "reals": state,
                        "fixed_time_horizon": t,
                        "reals_estimate": estimate * n,
                    }
                )

    # Convert to DataFrame
    result_df = pd.DataFrame(results)

    # Convert strata to categorical if needed
    result_df["strata"] = pd.Categorical(result_df["strata"])

    return result_df


def add_cutoff_strata(data: pl.DataFrame, by: float, stratified_by) -> pl.DataFrame:
    def transform_group(group: pl.DataFrame) -> pl.DataFrame:
        probs = group["probs"].to_numpy()
        columns_to_add = []

        if "probability_threshold" in stratified_by:
            breaks = create_breaks_values(probs, "probability_threshold", by)
            last_bin_index = len(breaks) - 2

            bin_indices = np.digitize(probs, bins=breaks, right=False) - 1
            bin_indices = np.where(probs == 1.0, last_bin_index, bin_indices)

            lower_bounds = breaks[bin_indices]
            upper_bounds = breaks[bin_indices + 1]

            include_upper_bounds = bin_indices == last_bin_index

            strata_prob_labels = np.where(
                include_upper_bounds,
                [f"[{lo:.2f}, {hi:.2f}]" for lo, hi in zip(lower_bounds, upper_bounds)],
                [f"[{lo:.2f}, {hi:.2f})" for lo, hi in zip(lower_bounds, upper_bounds)],
            ).astype(str)

            columns_to_add.append(
                pl.Series("strata_probability_threshold", strata_prob_labels)
            )

        if "ppcr" in stratified_by:
            # --- Compute strata_ppcr as quantiles on -probs ---
            try:
                q = int(1 / by)
                quantile_edges = np.quantile(-probs, np.linspace(0, 1, q))
                strata_ppcr = np.digitize(-probs, quantile_edges, right=False)
                strata_ppcr = (strata_ppcr / (1 / by)).astype(str)
            except ValueError:
                strata_ppcr = np.array(["1"] * len(probs))  # fallback for small group

        return group.with_columns(columns_to_add)

    # Apply per-group transformation
    grouped = data.partition_by("reference_group", as_dict=True)
    transformed_groups = [transform_group(group) for group in grouped.values()]
    return pl.concat(transformed_groups)


def create_strata_combinations(stratified_by: str, by: float) -> pl.DataFrame:
    breaks = create_breaks_values(None, "probability_threshold", by)

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
        lower_bound = strata_mid - by
        upper_bound = strata_mid + by
        mid_point = upper_bound - by
        include_lower_bound = np.ones_like(strata_mid, dtype=bool)
        include_upper_bound = np.zeros_like(strata_mid, dtype=bool)
        # chosen_cutoff = strata_mid
        strata = np.round(mid_point, 3).astype(str)
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


def create_aj_data_combinations(
    reference_groups: Sequence[str],
    fixed_time_horizons: Sequence[float],
    stratified_by: Sequence[str],
    by: float,
) -> pl.DataFrame:
    dfs = [create_strata_combinations(sb, by) for sb in stratified_by]
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

    # Define values for Cartesian product
    reals_labels = [
        "real_negatives",
        "real_positives",
        "real_competing",
        "real_censored",
    ]

    censoring_assumptions_labels = ["excluded", "adjusted"]

    competing_assumptions_labels = [
        "excluded",
        "adjusted_as_negative",
        "adjusted_as_censored",
        "adjusted_as_composite",
    ]

    combinations_frames: list[pl.DataFrame] = [
        _enum_dataframe("reference_group", reference_groups),
        pl.DataFrame(
            {"fixed_time_horizon": pl.Series(fixed_time_horizons, dtype=pl.Float64)}
        ),
        _enum_dataframe("censoring_assumption", censoring_assumptions_labels),
        _enum_dataframe("competing_assumption", competing_assumptions_labels),
        strata_combinations,
        _enum_dataframe("reals_labels", reals_labels),
    ]

    result = combinations_frames[0]
    for frame in combinations_frames[1:]:
        result = result.join(frame, how="cross")

    return result


def pivot_longer_strata(data: pl.DataFrame) -> pl.DataFrame:
    # Identify id_vars and value_vars
    id_vars = [col for col in data.columns if not col.startswith("strata_")]
    value_vars = [col for col in data.columns if col.startswith("strata_")]

    # Perform the melt (equivalent to pandas.melt)
    data_long = data.melt(
        id_vars=id_vars,
        value_vars=value_vars,
        variable_name="stratified_by",
        value_name="strata",
    )

    stratified_by_labels = ["probability_threshold", "ppcr"]
    stratified_by_enum = pl.Enum(stratified_by_labels)

    # Remove "strata_" prefix from the 'stratified_by' column
    data_long = data_long.with_columns(
        pl.col("stratified_by").str.replace("^strata_", "").cast(stratified_by_enum)
    )

    return data_long


def map_reals_to_labels_polars(data: pl.DataFrame) -> pl.DataFrame:
    return data.with_columns(
        [
            pl.when(pl.col("reals") == 0)
            .then("real_negatives")
            .when(pl.col("reals") == 1)
            .then("real_positives")
            .when(pl.col("reals") == 2)
            .then("real_competing")
            .otherwise("real_censored")
            .alias("reals")
        ]
    )


def update_administrative_censoring_polars(data: pl.DataFrame) -> pl.DataFrame:
    data = data.with_columns(
        [
            pl.when(
                (pl.col("times") > pl.col("fixed_time_horizon"))
                & (pl.col("reals_labels") == "real_positives")
            )
            .then(pl.lit("real_negatives"))
            .when(
                (pl.col("times") < pl.col("fixed_time_horizon"))
                & (pl.col("reals_labels") == "real_negatives")
            )
            .then(pl.lit("real_censored"))
            .otherwise(pl.col("reals_labels"))
            .alias("reals_labels")
        ]
    )

    return data


def create_aj_data(
    reference_group_data,
    censoring_heuristic,
    competing_heuristic,
    fixed_time_horizons,
    full_event_table: bool = False,
):
    """
    Create AJ estimates per strata based on censoring and competing assumptions.
    """

    def aj_estimates_with_cross(df, extra_cols):
        return df.join(pl.DataFrame(extra_cols), how="cross")

    exploded = assign_and_explode_polars(reference_group_data, fixed_time_horizons)

    event_table = prepare_event_table(reference_group_data)

    # TODO: solve strata in the pipeline

    excluded_events = _extract_excluded_events(
        event_table, fixed_time_horizons, censoring_heuristic, competing_heuristic
    )

    aj_df = _aj_adjusted_events(
        reference_group_data,
        exploded,
        censoring_heuristic,
        competing_heuristic,
        fixed_time_horizons,
        full_event_table,
    )

    result = aj_df.join(excluded_events, on=["fixed_time_horizon"], how="left")

    return aj_estimates_with_cross(
        result,
        {
            "censoring_assumption": censoring_heuristic,
            "competing_assumption": competing_heuristic,
        },
    ).select(
        [
            "strata",
            "fixed_time_horizon",
            "times",
            "real_negatives_est",
            "real_positives_est",
            "real_competing_est",
            "real_censored_est",
            "censoring_assumption",
            "competing_assumption",
            "estimate_origin",
        ]
    )


def _extract_excluded_events(
    event_table: pl.DataFrame,
    fixed_time_horizons: list[float],
    censoring_heuristic: str,
    competing_heuristic: str,
) -> pl.DataFrame:
    horizons_df = pl.DataFrame({"times": fixed_time_horizons}).sort("times")

    excluded_events = horizons_df.join_asof(
        event_table.with_columns(
            pl.col("count_0").cum_sum().cast(pl.Float64).alias("real_censored_est"),
            pl.col("count_2").cum_sum().cast(pl.Float64).alias("real_competing_est"),
        ).select(
            pl.col("times"),
            pl.col("real_censored_est"),
            pl.col("real_competing_est"),
        ),
        left_on="times",
        right_on="times",
    ).with_columns([pl.col("times").alias("fixed_time_horizon")])

    if censoring_heuristic != "excluded":
        excluded_events = excluded_events.with_columns(
            pl.lit(0.0).alias("real_censored_est")
        )

    if competing_heuristic != "excluded":
        excluded_events = excluded_events.with_columns(
            pl.lit(0.0).alias("real_competing_est")
        )

    return excluded_events


def extract_crude_estimate_polars(data: pl.DataFrame) -> pl.DataFrame:
    all_combinations = data.select(["strata", "reals", "fixed_time_horizon"]).unique()

    counts = data.group_by(["strata", "reals", "fixed_time_horizon"]).agg(
        pl.count().alias("reals_estimate")
    )

    return all_combinations.join(
        counts, on=["strata", "reals", "fixed_time_horizon"], how="left"
    ).with_columns([pl.col("reals_estimate").fill_null(0).cast(pl.Int64)])


# def update_administrative_censoring(data_to_adjust: pd.DataFrame) -> pd.DataFrame:
#     pl_df = pl.from_pandas(data_to_adjust)

#     # Perform the transformation using polars
#     pl_result = pl_df.with_columns(
#         pl.when(
#             (pl.col("times") > pl.col("fixed_time_horizon")) &
#             (pl.col("reals") == "real_positives")
#         ).then(
#             "real_negatives"
#         ).when(
#             (pl.col("times") < pl.col("fixed_time_horizon")) &
#             (pl.col("reals") == "real_negatives")
#         ).then(
#             "real_censored"
#         ).otherwise(
#             pl.col("reals")
#         ).alias("reals")
#     )

#     # Convert back to pandas DataFrame and return
#     result_pandas = pl_result.to_pandas()

#     return result_pandas


def extract_aj_estimate_for_strata(data_to_adjust, horizons, full_event_table: bool):
    n = data_to_adjust.height

    event_table = prepare_event_table(data_to_adjust)

    aj_estimate_for_strata_polars = predict_aj_estimates(
        event_table, pl.Series(horizons), full_event_table
    )

    if len(horizons) == 1:
        aj_estimate_for_strata_polars = aj_estimate_for_strata_polars.with_columns(
            pl.lit(horizons[0]).alias("fixed_time_horizon")
        )

    else:
        fixed_df = aj_estimate_for_strata_polars.filter(
            pl.col("estimate_origin") == "fixed_time_horizons"
        ).with_columns([pl.col("times").alias("fixed_time_horizon")])

        event_df = (
            aj_estimate_for_strata_polars.filter(
                pl.col("estimate_origin") == "event_table"
            )
            .with_columns([pl.lit(horizons).alias("fixed_time_horizon")])
            .explode("fixed_time_horizon")
        )

        aj_estimate_for_strata_polars = pl.concat(
            [fixed_df, event_df], how="vertical"
        ).sort("estimate_origin", "fixed_time_horizon", "times")

    return aj_estimate_for_strata_polars.with_columns(
        [
            (pl.col("state_occupancy_probability_0") * n).alias("real_negatives_est"),
            (pl.col("state_occupancy_probability_1") * n).alias("real_positives_est"),
            (pl.col("state_occupancy_probability_2") * n).alias("real_competing_est"),
            pl.col("fixed_time_horizon").cast(pl.Float64),
            pl.lit(data_to_adjust["strata"][0]).alias("strata"),
        ]
    ).select(
        [
            "strata",
            "times",
            "fixed_time_horizon",
            "real_negatives_est",
            "real_positives_est",
            "real_competing_est",
            pl.col("estimate_origin"),
        ]
    )


def assign_and_explode_polars(
    data: pl.DataFrame, fixed_time_horizons: list[float]
) -> pl.DataFrame:
    return (
        data.with_columns(pl.lit(fixed_time_horizons).alias("fixed_time_horizon"))
        .explode("fixed_time_horizon")
        .with_columns(pl.col("fixed_time_horizon").cast(pl.Float64))
    )


def create_list_data_to_adjust(
    probs_dict: Dict[str, np.ndarray],
    reals_dict: Union[np.ndarray, Dict[str, np.ndarray]],
    times_dict: Union[np.ndarray, Dict[str, np.ndarray]],
    stratified_by,
    by,
):
    # reference_groups = list(probs_dict.keys())
    reference_group_labels = list(probs_dict.keys())
    num_reals = len(reals_dict)

    reference_group_enum = pl.Enum(reference_group_labels)

    # Flatten and ensure list format
    data_to_adjust = pl.DataFrame(
        {
            "reference_group": np.repeat(reference_group_labels, num_reals),
            "probs": np.concatenate(
                [probs_dict[group] for group in reference_group_labels]
            ),
            "reals": np.tile(np.asarray(reals_dict), len(reference_group_labels)),
            "times": np.tile(np.asarray(times_dict), len(reference_group_labels)),
        }
    ).with_columns(pl.col("reference_group").cast(reference_group_enum))

    # Apply strata
    data_to_adjust = add_cutoff_strata(
        data_to_adjust, by=by, stratified_by=stratified_by
    )
    data_to_adjust = pivot_longer_strata(data_to_adjust)

    reals_labels = [
        "real_negatives",
        "real_positives",
        "real_competing",
        "real_censored",
    ]
    reals_enum = pl.Enum(reals_labels)

    # Map reals values to strings
    reals_map = {0: "real_negatives", 2: "real_competing", 1: "real_positives"}

    data_to_adjust = data_to_adjust.with_columns(
        pl.col("reals")
        .replace_strict(reals_map, return_dtype=reals_enum)
        .alias("reals_labels")
    )

    # Partition by reference_group
    list_data_to_adjust = {
        group[0]: df
        for group, df in data_to_adjust.partition_by(
            "reference_group", as_dict=True
        ).items()
    }

    return list_data_to_adjust


def ensure_no_categorical(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.select_dtypes(include="category").columns:
        df[col] = df[col].astype(str)
    return df


def extract_aj_estimate_by_assumptions(
    df: pl.DataFrame,
    assumptions_sets: list[dict],
    fixed_time_horizons: list[float],
) -> pl.DataFrame:
    aj_dfs = []

    for assumption in assumptions_sets:
        censoring = assumption["censoring_assumption"]
        competing = assumption["competing_assumption"]

        aj_df = create_aj_data(
            df, censoring, competing, fixed_time_horizons
        ).with_columns(
            [
                pl.lit(censoring).alias("censoring_assumption"),
                pl.lit(competing).alias("competing_assumption"),
            ]
        )

        aj_dfs.append(aj_df)

    aj_estimates_data = pl.concat(aj_dfs).drop(["estimate_origin", "times"])

    aj_estimates_unpivoted = aj_estimates_data.unpivot(
        index=[
            "strata",
            "fixed_time_horizon",
            "censoring_assumption",
            "competing_assumption",
        ],
        variable_name="reals_labels",
        value_name="reals_estimate",
    )

    return aj_estimates_unpivoted


def create_adjusted_data(
    list_data_to_adjust_polars: dict[str, pl.DataFrame],
    assumptions_sets: list[dict[str, str]],
    fixed_time_horizons: list[float],
) -> pl.DataFrame:
    all_results = []

    reference_groups = list(list_data_to_adjust_polars.keys())
    reference_group_enum = pl.Enum(reference_groups)

    censoring_assumption_labels = ["excluded", "adjusted"]
    censoring_assumption_enum = pl.Enum(censoring_assumption_labels)

    competing_assumption_labels = [
        "excluded",
        "adjusted_as_negative",
        "adjusted_as_censored",
        "adjusted_as_composite",
    ]
    competing_assumption_enum = pl.Enum(competing_assumption_labels)

    for reference_group, df in list_data_to_adjust_polars.items():
        input_df = df.select(["strata", "reals", "times"])

        aj_result = extract_aj_estimate_by_assumptions(
            input_df,
            assumptions_sets=assumptions_sets,
            fixed_time_horizons=fixed_time_horizons,
        )

        aj_result_with_group = aj_result.with_columns(
            [
                pl.lit(reference_group)
                .cast(reference_group_enum)
                .alias("reference_group")
            ]
        )

        all_results.append(aj_result_with_group)

    reals_enum_dtype = pl.Enum(
        [
            "real_negatives",
            "real_positives",
            "real_competing",
            "real_censored",
        ]
    )

    return (
        pl.concat(all_results)
        .with_columns([pl.col("reference_group").cast(reference_group_enum)])
        .with_columns(
            [
                pl.col("reals_labels").str.replace(r"_est$", "").cast(reals_enum_dtype),
                pl.col("censoring_assumption").cast(censoring_assumption_enum),
                pl.col("competing_assumption").cast(competing_assumption_enum),
            ]
        )
    )


def cast_and_join_adjusted_data(aj_data_combinations, aj_estimates_data):
    strata_enum_dtype = aj_data_combinations.schema["strata"]

    aj_estimates_data = aj_estimates_data.with_columns([pl.col("strata")]).with_columns(
        pl.col("strata").cast(strata_enum_dtype)
    )

    final_adjusted_data_polars = aj_data_combinations.with_columns(
        [pl.col("strata")]
    ).join(
        aj_estimates_data,
        on=[
            "strata",
            "fixed_time_horizon",
            "censoring_assumption",
            "competing_assumption",
            "reals_labels",
            "reference_group",
        ],
        how="left",
    )
    return final_adjusted_data_polars


def _censored_count(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.with_columns(
            ((pl.col("times") <= pl.col("fixed_time_horizon")) & (pl.col("reals") == 0))
            .cast(pl.Float64)
            .alias("is_censored")
        )
        .group_by(["strata", "fixed_time_horizon"])
        .agg(pl.col("is_censored").sum().alias("real_censored_est"))
    )


def _competing_count(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.with_columns(
            ((pl.col("times") <= pl.col("fixed_time_horizon")) & (pl.col("reals") == 2))
            .cast(pl.Float64)
            .alias("is_competing")
        )
        .group_by(["strata", "fixed_time_horizon"])
        .agg(pl.col("is_competing").sum().alias("real_competing_est"))
    )


def _aj_estimates_per_horizon(
    df: pl.DataFrame, horizons: list[float], full_event_table: bool
) -> pl.DataFrame:
    return pl.concat(
        [
            df.filter(pl.col("fixed_time_horizon") == h)
            .group_by("strata")
            .map_groups(
                lambda group: extract_aj_estimate_for_strata(
                    group, [h], full_event_table
                )
            )
            for h in horizons
        ],
        how="vertical",
    )


def _aj_adjusted_events(
    reference_group_data: pl.DataFrame,
    exploded: pl.DataFrame,
    censoring: str,
    competing: str,
    horizons: list[float],
    full_event_table: bool = False,
) -> pl.DataFrame:
    if censoring == "adjusted" and competing == "adjusted_as_negative":
        return reference_group_data.group_by("strata").map_groups(
            lambda group: extract_aj_estimate_for_strata(
                group, horizons, full_event_table
            )
        )

    if censoring == "excluded" and competing == "adjusted_as_negative":
        non_censored = exploded.filter(
            (pl.col("times") > pl.col("fixed_time_horizon")) | (pl.col("reals") > 0)
        )
        return _aj_estimates_per_horizon(non_censored, horizons, full_event_table)

    if censoring == "adjusted" and competing == "adjusted_as_censored":
        adjusted = reference_group_data.with_columns(
            pl.when(pl.col("reals") == 2)
            .then(pl.lit(0))
            .otherwise(pl.col("reals"))
            .alias("reals")
        )
        return adjusted.group_by("strata").map_groups(
            lambda group: extract_aj_estimate_for_strata(
                group, horizons, full_event_table
            )
        )

    if censoring == "excluded" and competing == "adjusted_as_censored":
        non_censored = exploded.filter(
            (pl.col("times") > pl.col("fixed_time_horizon")) | (pl.col("reals") > 0)
        ).with_columns(
            pl.when(pl.col("reals") == 2)
            .then(pl.lit(0))
            .otherwise(pl.col("reals"))
            .alias("reals")
        )
        return _aj_estimates_per_horizon(non_censored, horizons, full_event_table)

    if censoring == "adjusted" and competing == "adjusted_as_composite":
        adjusted = reference_group_data.with_columns(
            pl.when(pl.col("reals") == 2)
            .then(pl.lit(1))
            .otherwise(pl.col("reals"))
            .alias("reals")
        )
        return adjusted.group_by("strata").map_groups(
            lambda group: extract_aj_estimate_for_strata(
                group, horizons, full_event_table
            )
        )

    if censoring == "excluded" and competing == "adjusted_as_composite":
        non_censored = exploded.filter(
            (pl.col("times") > pl.col("fixed_time_horizon")) | (pl.col("reals") > 0)
        ).with_columns(
            pl.when(pl.col("reals") == 2)
            .then(pl.lit(1))
            .otherwise(pl.col("reals"))
            .alias("reals")
        )
        return _aj_estimates_per_horizon(non_censored, horizons, full_event_table)

    if censoring == "adjusted" and competing == "excluded":
        non_competing = exploded.filter(
            (pl.col("times") > pl.col("fixed_time_horizon")) | (pl.col("reals") != 2)
        ).with_columns(
            pl.when(pl.col("reals") == 2)
            .then(pl.lit(0))
            .otherwise(pl.col("reals"))
            .alias("reals")
        )
        return _aj_estimates_per_horizon(
            non_competing, horizons, full_event_table
        ).select(pl.exclude("real_competing_est"))

    # censoring == "excluded" and competing == "excluded"
    non_censored_non_competing = exploded.filter(
        (pl.col("times") > pl.col("fixed_time_horizon")) | (pl.col("reals") == 1)
    )

    return _aj_estimates_per_horizon(
        non_censored_non_competing, horizons, full_event_table
    ).drop("real_competing_est")

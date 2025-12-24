import pandas as pd
import polars as pl
from polarstate import predict_aj_estimates
from polarstate import prepare_event_table
from collections.abc import Sequence
from rtichoke.processing.transforms import assign_and_explode_polars


def create_aj_data(
    reference_group_data,
    breaks,
    censoring_heuristic,
    competing_heuristic,
    fixed_time_horizons,
    stratified_by: Sequence[str],
    full_event_table: bool = False,
    risk_set_scope: Sequence[str] = ["within_stratum"],
):
    """
    Create AJ estimates per strata based on censoring and competing heuristicss.
    """

    def aj_estimates_with_cross(df, extra_cols):
        return df.join(pl.DataFrame(extra_cols), how="cross")

    exploded = assign_and_explode_polars(reference_group_data, fixed_time_horizons)

    event_table = prepare_event_table(reference_group_data)

    # TODO: solve strata in the pipeline

    excluded_events = _extract_excluded_events(
        event_table, fixed_time_horizons, censoring_heuristic, competing_heuristic
    )

    aj_dfs = []
    for rscope in risk_set_scope:
        aj_res = _aj_adjusted_events(
            reference_group_data,
            breaks,
            exploded,
            censoring_heuristic,
            competing_heuristic,
            fixed_time_horizons,
            stratified_by,
            full_event_table,
            rscope,
        )

        aj_res = aj_res.select(
            [
                "strata",
                "times",
                "chosen_cutoff",
                "real_negatives_est",
                "real_positives_est",
                "real_competing_est",
                "estimate_origin",
                "fixed_time_horizon",
                "risk_set_scope",
            ]
        )

        aj_dfs.append(aj_res)

    aj_df = pl.concat(aj_dfs, how="vertical")

    result = aj_df.join(excluded_events, on=["fixed_time_horizon"], how="left")

    return aj_estimates_with_cross(
        result,
        {
            "censoring_heuristic": censoring_heuristic,
            "competing_heuristic": competing_heuristic,
        },
    ).select(
        [
            "strata",
            "chosen_cutoff",
            "fixed_time_horizon",
            "times",
            "real_negatives_est",
            "real_positives_est",
            "real_competing_est",
            "real_censored_est",
            "censoring_heuristic",
            "competing_heuristic",
            "estimate_origin",
            "risk_set_scope",
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


def extract_aj_estimate_by_cutoffs(
    data_to_adjust, horizons, breaks, stratified_by, full_event_table: bool
):
    # n = data_to_adjust.height

    counts_per_strata = (
        data_to_adjust.group_by(
            ["strata", "stratified_by", "upper_bound", "lower_bound"]
        )
        .len(name="strata_count")
        .with_columns(pl.col("strata_count").cast(pl.Float64))
    )

    aj_estimates_predicted_positives = pl.DataFrame()
    aj_estimates_predicted_negatives = pl.DataFrame()

    for stratification_criteria in stratified_by:
        for chosen_cutoff in breaks:
            if stratification_criteria == "probability_threshold":
                mask_predicted_positives = (pl.col("upper_bound") > chosen_cutoff) & (
                    pl.col("stratified_by") == "probability_threshold"
                )
                mask_predicted_negatives = (pl.col("upper_bound") <= chosen_cutoff) & (
                    pl.col("stratified_by") == "probability_threshold"
                )

            elif stratification_criteria == "ppcr":
                mask_predicted_positives = (
                    pl.col("lower_bound") > 1 - chosen_cutoff
                ) & (pl.col("stratified_by") == "ppcr")
                mask_predicted_negatives = (
                    pl.col("lower_bound") <= 1 - chosen_cutoff
                ) & (pl.col("stratified_by") == "ppcr")

            predicted_positives = data_to_adjust.filter(mask_predicted_positives)
            predicted_negatives = data_to_adjust.filter(mask_predicted_negatives)

            counts_per_strata_predicted_positives = counts_per_strata.filter(
                mask_predicted_positives
            )
            counts_per_strata_predicted_negatives = counts_per_strata.filter(
                mask_predicted_negatives
            )

            event_table_predicted_positives = prepare_event_table(predicted_positives)
            event_table_predicted_negatives = prepare_event_table(predicted_negatives)

            aj_estimate_predicted_positives = (
                (
                    predict_aj_estimates(
                        event_table_predicted_positives,
                        pl.Series(horizons),
                        full_event_table,
                    )
                    .with_columns(
                        pl.lit(chosen_cutoff).alias("chosen_cutoff"),
                        pl.lit(stratification_criteria)
                        .alias("stratified_by")
                        .cast(pl.Enum(["probability_threshold", "ppcr"])),
                    )
                    .join(
                        counts_per_strata_predicted_positives,
                        on=["stratified_by"],
                        how="left",
                    )
                    .with_columns(
                        [
                            (
                                pl.col("state_occupancy_probability_0")
                                * pl.col("strata_count")
                            ).alias("real_negatives_est"),
                            (
                                pl.col("state_occupancy_probability_1")
                                * pl.col("strata_count")
                            ).alias("real_positives_est"),
                            (
                                pl.col("state_occupancy_probability_2")
                                * pl.col("strata_count")
                            ).alias("real_competing_est"),
                        ]
                    )
                )
                .select(
                    [
                        "strata",
                        # "stratified_by",
                        "times",
                        "chosen_cutoff",
                        "real_negatives_est",
                        "real_positives_est",
                        "real_competing_est",
                        "estimate_origin",
                    ]
                )
                .with_columns([pl.col("times").alias("fixed_time_horizon")])
            )

            aj_estimate_predicted_negatives = (
                (
                    predict_aj_estimates(
                        event_table_predicted_negatives,
                        pl.Series(horizons),
                        full_event_table,
                    )
                    .with_columns(
                        pl.lit(chosen_cutoff).alias("chosen_cutoff"),
                        pl.lit(stratification_criteria)
                        .alias("stratified_by")
                        .cast(pl.Enum(["probability_threshold", "ppcr"])),
                    )
                    .join(
                        counts_per_strata_predicted_negatives,
                        on=["stratified_by"],
                        how="left",
                    )
                    .with_columns(
                        [
                            (
                                pl.col("state_occupancy_probability_0")
                                * pl.col("strata_count")
                            ).alias("real_negatives_est"),
                            (
                                pl.col("state_occupancy_probability_1")
                                * pl.col("strata_count")
                            ).alias("real_positives_est"),
                            (
                                pl.col("state_occupancy_probability_2")
                                * pl.col("strata_count")
                            ).alias("real_competing_est"),
                        ]
                    )
                )
                .select(
                    [
                        "strata",
                        # "stratified_by",
                        "times",
                        "chosen_cutoff",
                        "real_negatives_est",
                        "real_positives_est",
                        "real_competing_est",
                        "estimate_origin",
                    ]
                )
                .with_columns([pl.col("times").alias("fixed_time_horizon")])
            )

            aj_estimates_predicted_negatives = pl.concat(
                [aj_estimates_predicted_negatives, aj_estimate_predicted_negatives],
                how="vertical",
            )

            aj_estimates_predicted_positives = pl.concat(
                [aj_estimates_predicted_positives, aj_estimate_predicted_positives],
                how="vertical",
            )

    aj_estimate_by_cutoffs = pl.concat(
        [aj_estimates_predicted_negatives, aj_estimates_predicted_positives],
        how="vertical",
    )

    return aj_estimate_by_cutoffs


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


def ensure_no_categorical(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.select_dtypes(include="category").columns:
        df[col] = df[col].astype(str)
    return df


def extract_aj_estimate_by_heuristics(
    df: pl.DataFrame,
    breaks: Sequence[float],
    heuristics_sets: list[dict],
    fixed_time_horizons: list[float],
    stratified_by: Sequence[str],
    risk_set_scope: Sequence[str] = ["within_stratum"],
) -> pl.DataFrame:
    aj_dfs = []

    for heuristic in heuristics_sets:
        censoring = heuristic["censoring_heuristic"]
        competing = heuristic["competing_heuristic"]

        aj_df = create_aj_data(
            df,
            breaks,
            censoring,
            competing,
            fixed_time_horizons,
            stratified_by=stratified_by,
            full_event_table=False,
            risk_set_scope=risk_set_scope,
        ).with_columns(
            [
                pl.lit(censoring).alias("censoring_heuristic"),
                pl.lit(competing).alias("competing_heuristic"),
            ]
        )

        aj_dfs.append(aj_df)

    aj_estimates_data = pl.concat(aj_dfs).drop(["estimate_origin", "times"])

    aj_estimates_unpivoted = aj_estimates_data.unpivot(
        index=[
            "strata",
            "chosen_cutoff",
            "fixed_time_horizon",
            "censoring_heuristic",
            "competing_heuristic",
            "risk_set_scope",
        ],
        variable_name="reals_labels",
        value_name="reals_estimate",
    )

    return aj_estimates_unpivoted


def _create_adjusted_data_binary(
    list_data_to_adjust: dict[str, pl.DataFrame],
    breaks: Sequence[float],
    stratified_by: Sequence[str],
) -> pl.DataFrame:
    long_df = pl.concat(list(list_data_to_adjust.values()), how="vertical")

    adjusted_data_binary = (
        long_df.group_by(["strata", "stratified_by", "reference_group", "reals_labels"])
        .agg(pl.count().alias("reals_estimate"))
        .join(pl.DataFrame({"chosen_cutoff": breaks}), how="cross")
    )

    return adjusted_data_binary


def create_adjusted_data(
    list_data_to_adjust: dict[str, pl.DataFrame],
    heuristics_sets: list[dict[str, str]],
    fixed_time_horizons: list[float],
    breaks: Sequence[float],
    stratified_by: Sequence[str],
    risk_set_scope: Sequence[str] = ["within_stratum"],
) -> pl.DataFrame:
    all_results = []

    reference_groups = list(list_data_to_adjust.keys())
    reference_group_enum = pl.Enum(reference_groups)

    heuristics_df = pl.DataFrame(heuristics_sets)
    censoring_heuristic_enum = pl.Enum(
        heuristics_df["censoring_heuristic"].unique(maintain_order=True)
    )
    competing_heuristic_enum = pl.Enum(
        heuristics_df["competing_heuristic"].unique(maintain_order=True)
    )

    for reference_group, df in list_data_to_adjust.items():
        input_df = df.select(
            ["strata", "reals", "times", "upper_bound", "lower_bound", "stratified_by"]
        )

        aj_result = extract_aj_estimate_by_heuristics(
            input_df,
            breaks,
            heuristics_sets=heuristics_sets,
            fixed_time_horizons=fixed_time_horizons,
            stratified_by=stratified_by,
            risk_set_scope=risk_set_scope,
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
                pl.col("censoring_heuristic").cast(censoring_heuristic_enum),
                pl.col("competing_heuristic").cast(competing_heuristic_enum),
            ]
        )
    )


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


def _aj_estimates_by_cutoff_per_horizon(
    df: pl.DataFrame,
    horizons: list[float],
    breaks: Sequence[float],
    stratified_by: Sequence[str],
) -> pl.DataFrame:
    return pl.concat(
        [
            df.filter(pl.col("fixed_time_horizon") == h)
            .group_by("strata")
            .map_groups(
                lambda group: extract_aj_estimate_by_cutoffs(
                    group, [h], breaks, stratified_by, full_event_table=False
                )
            )
            for h in horizons
        ],
        how="vertical",
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
    breaks: Sequence[float],
    exploded: pl.DataFrame,
    censoring: str,
    competing: str,
    horizons: list[float],
    stratified_by: Sequence[str],
    full_event_table: bool = False,
    risk_set_scope: Sequence[str] = ["within_stratum"],
) -> pl.DataFrame:
    strata_enum_dtype = reference_group_data.schema["strata"]

    # Special-case: adjusted censoring + competing adjusted_as_negative supports pooled_by_cutoff
    if censoring == "adjusted" and competing == "adjusted_as_negative":
        if risk_set_scope == "within_stratum":
            adjusted = (
                reference_group_data.group_by("strata")
                .map_groups(
                    lambda group: extract_aj_estimate_for_strata(
                        group, horizons, full_event_table
                    )
                )
                .join(pl.DataFrame({"chosen_cutoff": breaks}), how="cross")
            )
            # preserve the original enum dtype for 'strata' coming from reference_group_data

            adjusted = adjusted.with_columns(
                [
                    pl.col("strata").cast(strata_enum_dtype),
                    pl.lit(risk_set_scope)
                    .cast(pl.Enum(["within_stratum", "pooled_by_cutoff"]))
                    .alias("risk_set_scope"),
                ]
            )

            return adjusted

        elif risk_set_scope == "pooled_by_cutoff":
            adjusted = extract_aj_estimate_by_cutoffs(
                reference_group_data, horizons, breaks, stratified_by, full_event_table
            )
            adjusted = adjusted.with_columns(
                pl.lit(risk_set_scope)
                .cast(pl.Enum(["within_stratum", "pooled_by_cutoff"]))
                .alias("risk_set_scope")
            )
            return adjusted

    # Special-case: both excluded (faster branch in original)
    if censoring == "excluded" and competing == "excluded":
        non_censored_non_competing = exploded.filter(
            (pl.col("times") > pl.col("fixed_time_horizon")) | (pl.col("reals") == 1)
        )

        adjusted = _aj_estimates_per_horizon(
            non_censored_non_competing, horizons, full_event_table
        )

        adjusted = adjusted.with_columns(
            [
                pl.col("strata").cast(strata_enum_dtype),
                pl.lit(risk_set_scope)
                .cast(pl.Enum(["within_stratum", "pooled_by_cutoff"]))
                .alias("risk_set_scope"),
            ]
        ).join(pl.DataFrame({"chosen_cutoff": breaks}), how="cross")

        return adjusted

    # Special-case: competing excluded (handled by filtering out competing events)
    if competing == "excluded":
        # Use exploded to apply filters that depend on fixed_time_horizon consistently
        non_competing = exploded.filter(
            (pl.col("times") > pl.col("fixed_time_horizon")) | (pl.col("reals") != 2)
        ).with_columns(
            pl.when(pl.col("reals") == 2)
            .then(pl.lit(0))
            .otherwise(pl.col("reals"))
            .alias("reals")
        )

        if risk_set_scope == "within_stratum":
            adjusted = (
                _aj_estimates_per_horizon(non_competing, horizons, full_event_table)
                # .select(pl.exclude("real_competing_est"))
                .join(pl.DataFrame({"chosen_cutoff": breaks}), how="cross")
            )

        elif risk_set_scope == "pooled_by_cutoff":
            adjusted = extract_aj_estimate_by_cutoffs(
                non_competing, horizons, breaks, stratified_by, full_event_table
            )

        adjusted = adjusted.with_columns(
            [
                pl.col("strata").cast(strata_enum_dtype),
                pl.lit(risk_set_scope)
                .cast(pl.Enum(["within_stratum", "pooled_by_cutoff"]))
                .alias("risk_set_scope"),
            ]
        )
        return adjusted

    # For remaining cases, determine base dataframe depending on censoring rule:
    # - "adjusted": use the full reference_group_data (events censored at horizon are kept/adjusted)
    # - "excluded": remove administratively censored observations (use exploded with filter)
    base_df = (
        exploded.filter(
            (pl.col("times") > pl.col("fixed_time_horizon")) | (pl.col("reals") > 0)
        )
        if censoring == "excluded"
        else reference_group_data
    )

    # Apply competing-event transformation if required
    if competing == "adjusted_as_censored":
        base_df = base_df.with_columns(
            pl.when(pl.col("reals") == 2)
            .then(pl.lit(0))
            .otherwise(pl.col("reals"))
            .alias("reals")
        )
    elif competing == "adjusted_as_composite":
        base_df = base_df.with_columns(
            pl.when(pl.col("reals") == 2)
            .then(pl.lit(1))
            .otherwise(pl.col("reals"))
            .alias("reals")
        )
    # competing == "adjusted_as_negative": keep reals as-is (no transform)

    # Finally choose aggregation strategy: per-stratum or horizon-wise
    if censoring == "excluded":
        # For excluded censoring we always evaluate per-horizon on the filtered (exploded) dataset

        if risk_set_scope == "within_stratum":
            adjusted = _aj_estimates_per_horizon(base_df, horizons, full_event_table)

            adjusted = adjusted.join(
                pl.DataFrame({"chosen_cutoff": breaks}), how="cross"
            )

        elif risk_set_scope == "pooled_by_cutoff":
            adjusted = _aj_estimates_by_cutoff_per_horizon(
                base_df, horizons, breaks, stratified_by
            )

        adjusted = adjusted.with_columns(
            pl.lit(risk_set_scope)
            .cast(pl.Enum(["within_stratum", "pooled_by_cutoff"]))
            .alias("risk_set_scope")
        )

        return adjusted.with_columns(pl.col("strata").cast(strata_enum_dtype))
    else:
        # For adjusted censoring we aggregate within strata

        if risk_set_scope == "within_stratum":
            adjusted = (
                base_df.group_by("strata")
                .map_groups(
                    lambda group: extract_aj_estimate_for_strata(
                        group, horizons, full_event_table
                    )
                )
                .join(pl.DataFrame({"chosen_cutoff": breaks}), how="cross")
            )

        elif risk_set_scope == "pooled_by_cutoff":
            adjusted = extract_aj_estimate_by_cutoffs(
                base_df, horizons, breaks, stratified_by, full_event_table
            )

        adjusted = adjusted.with_columns(
            [
                pl.col("strata").cast(strata_enum_dtype),
                pl.lit(risk_set_scope)
                .cast(pl.Enum(["within_stratum", "pooled_by_cutoff"]))
                .alias("risk_set_scope"),
            ]
        )

        return adjusted

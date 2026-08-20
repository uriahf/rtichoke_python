import numpy as np
import polars as pl
from typing import Dict, Union
from rtichoke.processing.combinations import create_breaks_values


def add_cutoff_strata(data: pl.DataFrame, by: float, stratified_by) -> pl.DataFrame:
    def transform_group(group: pl.DataFrame, by: float) -> pl.DataFrame:
        probs = group["probs"].to_numpy()
        columns_to_add = []

        breaks = create_breaks_values(probs, "probability_threshold", by)
        if "probability_threshold" in stratified_by:
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
            # --- Compute strata_ppcr as equal-frequency quantile bins by rank ---
            by = float(by)
            q = int(round(1 / by))  # e.g. 0.2 -> 5 bins

            probs = np.asarray(probs, float)

            edges = np.quantile(probs, np.linspace(0.0, 1.0, q + 1), method="linear")

            edges = np.maximum.accumulate(edges)

            edges[0] = 0.0
            edges[-1] = 1.0

            bin_idx = np.digitize(probs, bins=edges[1:-1], right=True)

            s = str(by)
            decimals = len(s.split(".")[-1]) if "." in s else 0

            labels = [f"{x:.{decimals}f}" for x in np.linspace(by, 1.0, q)]

            strata_labels = np.array(labels)[bin_idx]

            columns_to_add.append(
                pl.Series("strata_ppcr", strata_labels).cast(pl.Enum(labels))
            )
        return group.with_columns(columns_to_add)

    # Apply per-group transformation
    grouped = data.partition_by("reference_group", as_dict=True)
    transformed_groups = [transform_group(group, by) for group in grouped.values()]
    return pl.concat(transformed_groups)


def pivot_longer_strata(data: pl.DataFrame) -> pl.DataFrame:
    # Identify id_vars and value_vars
    index_cols = [col for col in data.columns if not col.startswith("strata_")]
    on_cols = [col for col in data.columns if col.startswith("strata_")]

    # Perform the unpivot (equivalent to pandas.melt)
    data_long = data.unpivot(
        index=index_cols,
        on=on_cols,
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


def assign_and_explode_polars(
    data: pl.DataFrame, fixed_time_horizons: list[float]
) -> pl.DataFrame:
    return (
        data.with_columns(pl.lit(fixed_time_horizons).alias("fixed_time_horizon"))
        .explode("fixed_time_horizon")
        .with_columns(pl.col("fixed_time_horizon").cast(pl.Float64))
    )


def _create_list_data_to_adjust_binary(
    aj_data_combinations: pl.DataFrame,
    probs_dict: Dict[str, np.ndarray],
    reals_dict: Union[np.ndarray, Dict[str, np.ndarray]],
    stratified_by,
    by,
) -> Dict[str, pl.DataFrame]:
    reference_group_labels = list(probs_dict.keys())

    if isinstance(reals_dict, dict):
        num_keys_reals = len(reals_dict)
    else:
        num_keys_reals = 1

    reference_group_enum = pl.Enum(reference_group_labels)

    strata_enum_dtype = aj_data_combinations.schema["strata"]

    if len(probs_dict) == 1:
        probs_array = np.asarray(probs_dict[reference_group_labels[0]])

        data_to_adjust = pl.DataFrame(
            {
                "reference_group": np.repeat(reference_group_labels, len(probs_array)),
                "probs": probs_array,
                "reals": reals_dict,
            }
        ).with_columns(pl.col("reference_group").cast(reference_group_enum))

    elif num_keys_reals == 1:
        data_to_adjust = pl.DataFrame(
            {
                "reference_group": np.repeat(reference_group_labels, len(reals_dict)),
                "probs": np.concatenate(
                    [probs_dict[group] for group in reference_group_labels]
                ),
                "reals": np.tile(np.asarray(reals_dict), len(reference_group_labels)),
            }
        ).with_columns(pl.col("reference_group").cast(reference_group_enum))

    elif isinstance(reals_dict, dict):
        data_to_adjust = (
            pl.DataFrame(
                {
                    "reference_group": list(probs_dict.keys()),
                    "probs": list(probs_dict.values()),
                    "reals": list(reals_dict.values()),
                }
            )
            .explode(["probs", "reals"])
            .with_columns(pl.col("reference_group").cast(reference_group_enum))
        )

    data_to_adjust = add_cutoff_strata(
        data_to_adjust, by=by, stratified_by=stratified_by
    )

    data_to_adjust = pivot_longer_strata(data_to_adjust)

    data_to_adjust = (
        data_to_adjust.with_columns([pl.col("strata")])
        .with_columns(pl.col("strata").cast(strata_enum_dtype))
        .join(
            aj_data_combinations.select(
                pl.col("strata"),
                pl.col("stratified_by"),
                pl.col("upper_bound"),
                pl.col("lower_bound"),
            ).unique(),
            how="left",
            on=["strata", "stratified_by"],
        )
    )

    reals_labels = ["real_negatives", "real_positives"]

    reals_enum = pl.Enum(reals_labels)

    reals_map = {0: "real_negatives", 1: "real_positives"}

    data_to_adjust = data_to_adjust.with_columns(
        pl.col("reals")
        .replace_strict(reals_map, return_dtype=reals_enum)
        .alias("reals_labels")
    )

    list_data_to_adjust = {
        group[0]: df
        for group, df in data_to_adjust.partition_by(
            "reference_group", as_dict=True
        ).items()
    }

    return list_data_to_adjust


def _create_list_data_to_adjust(
    aj_data_combinations: pl.DataFrame,
    probs_dict: Dict[str, np.ndarray],
    reals_dict: Union[np.ndarray, Dict[str, np.ndarray]],
    times_dict: Union[np.ndarray, Dict[str, np.ndarray]],
    stratified_by,
    by,
) -> Dict[str, pl.DataFrame]:
    # reference_groups = list(probs_dict.keys())
    reference_group_labels = list(probs_dict.keys())

    if isinstance(reals_dict, dict):
        num_keys_reals = len(reals_dict)
    else:
        num_keys_reals = 1

    # num_reals = len(reals_dict)

    reference_group_enum = pl.Enum(reference_group_labels)

    strata_enum_dtype = aj_data_combinations.schema["strata"]

    if len(probs_dict) == 1:
        probs_array = np.asarray(probs_dict[reference_group_labels[0]])

        if isinstance(reals_dict, dict):
            reals_array = np.asarray(reals_dict[reference_group_labels[0]])
        else:
            reals_array = np.asarray(reals_dict)

        if isinstance(times_dict, dict):
            times_array = np.asarray(times_dict[reference_group_labels[0]])
        else:
            times_array = np.asarray(times_dict)

        data_to_adjust = pl.DataFrame(
            {
                "reference_group": np.repeat(reference_group_labels, len(probs_array)),
                "probs": probs_array,
                "reals": reals_array,
                "times": times_array,
            }
        ).with_columns(pl.col("reference_group").cast(reference_group_enum))

    elif num_keys_reals == 1:
        reals_array = np.asarray(reals_dict)
        times_array = np.asarray(times_dict)
        n = len(reals_array)

        data_to_adjust = pl.DataFrame(
            {
                "reference_group": np.repeat(reference_group_labels, n),
                "probs": np.concatenate(
                    [np.asarray(probs_dict[g]) for g in reference_group_labels]
                ),
                "reals": np.tile(reals_array, len(reference_group_labels)),
                "times": np.tile(times_array, len(reference_group_labels)),
            }
        ).with_columns(pl.col("reference_group").cast(reference_group_enum))

    elif isinstance(reals_dict, dict) and isinstance(times_dict, dict):
        data_to_adjust = (
            pl.DataFrame(
                {
                    "reference_group": reference_group_labels,
                    "probs": list(probs_dict.values()),
                    "reals": list(reals_dict.values()),
                    "times": list(times_dict.values()),
                }
            )
            .explode(["probs", "reals", "times"])
            .with_columns(pl.col("reference_group").cast(reference_group_enum))
        )

    data_to_adjust = add_cutoff_strata(
        data_to_adjust, by=by, stratified_by=stratified_by
    )

    data_to_adjust = pivot_longer_strata(data_to_adjust)

    data_to_adjust = (
        data_to_adjust.with_columns([pl.col("strata")])
        .with_columns(pl.col("strata").cast(strata_enum_dtype))
        .join(
            aj_data_combinations.select(
                pl.col("strata"),
                pl.col("stratified_by"),
                pl.col("upper_bound"),
                pl.col("lower_bound"),
            ).unique(),
            how="left",
            on=["strata", "stratified_by"],
        )
    )

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


def _cast_and_join_adjusted_data_binary(
    aj_data_combinations: pl.DataFrame, aj_estimates_data: pl.DataFrame
) -> pl.DataFrame:
    strata_enum_dtype = aj_data_combinations.schema["strata"]

    aj_estimates_data = aj_estimates_data.with_columns([pl.col("strata")]).with_columns(
        pl.col("strata").cast(strata_enum_dtype)
    )

    final_adjusted_data_polars = (
        (
            aj_data_combinations.with_columns([pl.col("strata")]).join(
                aj_estimates_data,
                on=[
                    "strata",
                    "stratified_by",
                    "reals_labels",
                    "reference_group",
                    "chosen_cutoff",
                ],
                how="left",
            )
        )
        .with_columns(
            pl.when(
                (
                    (pl.col("chosen_cutoff") >= pl.col("upper_bound"))
                    & (pl.col("stratified_by") == "probability_threshold")
                )
                | (
                    ((1 - pl.col("chosen_cutoff")) >= pl.col("mid_point"))
                    & (pl.col("stratified_by") == "ppcr")
                )
            )
            .then(pl.lit("predicted_negatives"))
            .otherwise(pl.lit("predicted_positives"))
            .cast(pl.Enum(["predicted_negatives", "predicted_positives"]))
            .alias("prediction_label")
        )
        .with_columns(
            (
                pl.when(
                    (pl.col("prediction_label") == pl.lit("predicted_positives"))
                    & (pl.col("reals_labels") == pl.lit("real_positives"))
                )
                .then(pl.lit("true_positives"))
                .when(
                    (pl.col("prediction_label") == pl.lit("predicted_positives"))
                    & (pl.col("reals_labels") == pl.lit("real_negatives"))
                )
                .then(pl.lit("false_positives"))
                .when(
                    (pl.col("prediction_label") == pl.lit("predicted_negatives"))
                    & (pl.col("reals_labels") == pl.lit("real_negatives"))
                )
                .then(pl.lit("true_negatives"))
                .when(
                    (pl.col("prediction_label") == pl.lit("predicted_negatives"))
                    & (pl.col("reals_labels") == pl.lit("real_positives"))
                )
                .then(pl.lit("false_negatives"))
                .cast(
                    pl.Enum(
                        [
                            "true_positives",
                            "false_positives",
                            "true_negatives",
                            "false_negatives",
                        ]
                    )
                )
            ).alias("classification_outcome")
        )
    ).with_columns(pl.col("reals_estimate").fill_null(0))

    return final_adjusted_data_polars


def cast_and_join_adjusted_data(
    aj_data_combinations, aj_estimates_data
) -> pl.DataFrame:
    strata_enum_dtype = aj_data_combinations.schema["strata"]

    aj_estimates_data = aj_estimates_data.with_columns([pl.col("strata")]).with_columns(
        pl.col("strata").cast(strata_enum_dtype)
    )

    final_adjusted_data_polars = (
        aj_data_combinations.with_columns([pl.col("strata")])
        .join(
            aj_estimates_data,
            on=[
                "strata",
                "fixed_time_horizon",
                "censoring_heuristic",
                "competing_heuristic",
                "reals_labels",
                "reference_group",
                "chosen_cutoff",
                "risk_set_scope",
            ],
            how="left",
        )
        .with_columns(
            pl.when(
                (
                    (pl.col("chosen_cutoff") >= pl.col("upper_bound"))
                    & (pl.col("stratified_by") == "probability_threshold")
                )
                | (
                    ((1 - pl.col("chosen_cutoff")) >= pl.col("mid_point"))
                    & (pl.col("stratified_by") == "ppcr")
                )
            )
            .then(pl.lit("predicted_negatives"))
            .otherwise(pl.lit("predicted_positives"))
            .cast(pl.Enum(["predicted_negatives", "predicted_positives"]))
            .alias("prediction_label")
        )
        .with_columns(
            (
                pl.when(
                    (pl.col("prediction_label") == pl.lit("predicted_positives"))
                    & (pl.col("reals_labels") == pl.lit("real_positives"))
                )
                .then(pl.lit("true_positives"))
                .when(
                    (pl.col("prediction_label") == pl.lit("predicted_positives"))
                    & (pl.col("reals_labels") == pl.lit("real_negatives"))
                )
                .then(pl.lit("false_positives"))
                .when(
                    (pl.col("prediction_label") == pl.lit("predicted_negatives"))
                    & (pl.col("reals_labels") == pl.lit("real_negatives"))
                )
                .then(pl.lit("true_negatives"))
                .when(
                    (pl.col("prediction_label") == pl.lit("predicted_negatives"))
                    & (pl.col("reals_labels") == pl.lit("real_positives"))
                )
                .then(pl.lit("false_negatives"))
                .when(
                    (pl.col("prediction_label") == pl.lit("predicted_negatives"))
                    & (pl.col("reals_labels") == pl.lit("real_competing"))
                    & (pl.col("competing_heuristic") == pl.lit("adjusted_as_negative"))
                )
                .then(pl.lit("true_negatives"))
                .when(
                    (pl.col("prediction_label") == pl.lit("predicted_positives"))
                    & (pl.col("reals_labels") == pl.lit("real_competing"))
                    & (pl.col("competing_heuristic") == pl.lit("adjusted_as_negative"))
                )
                .then(pl.lit("false_positives"))
                .otherwise(pl.lit("excluded"))  # or pl.lit(None) if you prefer nulls
                .cast(
                    pl.Enum(
                        [
                            "true_positives",
                            "false_positives",
                            "true_negatives",
                            "false_negatives",
                            "excluded",
                        ]
                    )
                )
            ).alias("classification_outcome")
        )
    )
    return final_adjusted_data_polars


def _calculate_cumulative_aj_data_binary(aj_data: pl.DataFrame) -> pl.DataFrame:
    cumulative_aj_data = (
        aj_data.group_by(
            [
                "reference_group",
                "stratified_by",
                "chosen_cutoff",
                "classification_outcome",
            ]
        )
        .agg([pl.col("reals_estimate").sum()])
        .pivot(on="classification_outcome", values="reals_estimate")
        .with_columns(
            [
                pl.col(col).fill_null(0)
                for col in [
                    "true_positives",
                    "true_negatives",
                    "false_positives",
                    "false_negatives",
                ]
            ]
        )
        .with_columns(
            (pl.col("true_positives") + pl.col("false_positives")).alias(
                "predicted_positives"
            ),
            (pl.col("true_negatives") + pl.col("false_negatives")).alias(
                "predicted_negatives"
            ),
            (pl.col("true_positives") + pl.col("false_negatives")).alias(
                "real_positives"
            ),
            (pl.col("false_positives") + pl.col("true_negatives")).alias(
                "real_negatives"
            ),
            (
                pl.col("true_positives")
                + pl.col("true_negatives")
                + pl.col("false_positives")
                + pl.col("false_negatives")
            )
            .alias("n")
            .sum(),
        )
        .with_columns(
            (pl.col("true_positives") + pl.col("false_positives")).alias(
                "predicted_positives"
            ),
            (pl.col("true_negatives") + pl.col("false_negatives")).alias(
                "predicted_negatives"
            ),
            (pl.col("true_positives") + pl.col("false_negatives")).alias(
                "real_positives"
            ),
            (pl.col("false_positives") + pl.col("true_negatives")).alias(
                "real_negatives"
            ),
            (
                pl.col("true_positives")
                + pl.col("true_negatives")
                + pl.col("false_positives")
                + pl.col("false_negatives")
            ).alias("n"),
        )
    )

    return cumulative_aj_data


def _calculate_cumulative_aj_data(aj_data: pl.DataFrame) -> pl.DataFrame:
    cumulative_aj_data = (
        aj_data.filter(pl.col("risk_set_scope") == "pooled_by_cutoff")
        .group_by(
            [
                "reference_group",
                "fixed_time_horizon",
                "censoring_heuristic",
                "competing_heuristic",
                "stratified_by",
                "chosen_cutoff",
                "classification_outcome",
            ]
        )
        .agg([pl.col("reals_estimate").sum()])
        .pivot(on="classification_outcome", values="reals_estimate")
        .fill_null(0)
        .with_columns(
            (pl.col("true_positives") + pl.col("false_positives")).alias(
                "predicted_positives"
            ),
            (pl.col("true_negatives") + pl.col("false_negatives")).alias(
                "predicted_negatives"
            ),
            (pl.col("true_positives") + pl.col("false_negatives")).alias(
                "real_positives"
            ),
            (pl.col("false_positives") + pl.col("true_negatives")).alias(
                "real_negatives"
            ),
            (
                pl.col("true_positives")
                + pl.col("true_negatives")
                + pl.col("false_positives")
                + pl.col("false_negatives")
            ).alias("n"),
        )
        .with_columns(
            (pl.col("true_positives") + pl.col("false_positives")).alias(
                "predicted_positives"
            ),
            (pl.col("true_negatives") + pl.col("false_negatives")).alias(
                "predicted_negatives"
            ),
            (pl.col("true_positives") + pl.col("false_negatives")).alias(
                "real_positives"
            ),
            (pl.col("false_positives") + pl.col("true_negatives")).alias(
                "real_negatives"
            ),
            (
                pl.col("true_positives")
                + pl.col("true_negatives")
                + pl.col("false_positives")
                + pl.col("false_negatives")
            ).alias("n"),
        )
    )

    return cumulative_aj_data


def _turn_cumulative_aj_to_performance_data(
    cumulative_aj_data: pl.DataFrame,
) -> pl.DataFrame:
    performance_data = cumulative_aj_data.with_columns(
        (pl.col("true_positives") / pl.col("real_positives")).alias("sensitivity"),
        (pl.col("true_negatives") / pl.col("real_negatives")).alias("specificity"),
        (pl.col("true_positives") / pl.col("predicted_positives")).alias("ppv"),
        (pl.col("true_negatives") / pl.col("predicted_negatives")).alias("npv"),
        (pl.col("false_positives") / pl.col("real_negatives")).alias(
            "false_positive_rate"
        ),
        (
            (pl.col("true_positives") / pl.col("predicted_positives"))
            / (pl.col("real_positives") / pl.col("n"))
        ).alias("lift"),
        pl.when(pl.col("stratified_by") == "probability_threshold")
        .then(
            (pl.col("true_positives") / pl.col("n"))
            - (pl.col("false_positives") / pl.col("n"))
            * pl.col("chosen_cutoff")
            / (1 - pl.col("chosen_cutoff"))
        )
        .otherwise(None)
        .alias("net_benefit"),
        pl.when(pl.col("stratified_by") == "probability_threshold")
        .then(
            100 * (pl.col("true_negatives") / pl.col("n"))
            - (pl.col("false_negatives") / pl.col("n"))
            * (1 - pl.col("chosen_cutoff"))
            / pl.col("chosen_cutoff")
        )
        .otherwise(None)
        .alias("net_benefit_interventions_avoided"),
        pl.when(pl.col("stratified_by") == "probability_threshold")
        .then(pl.col("predicted_positives") / pl.col("n"))
        .otherwise(pl.col("chosen_cutoff"))
        .alias("ppcr"),
    )

    return performance_data

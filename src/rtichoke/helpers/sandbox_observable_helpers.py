from lifelines import AalenJohansenFitter
import pandas as pd
import numpy as np
import itertools
import polars as pl

print("Polars version:", pl.__version__)


# def extract_aj_estimate(data_to_adjust, fixed_time_horizons) -> pd.DataFrame:
#     """
#     Hybrid version that accepts either pandas or Polars DataFrame and applies
#     Aalen-Johansen estimation using lifelines (pandas-based).
#     """

#     # Ensure input is a pandas DataFrame
#     if isinstance(data_to_adjust, pl.DataFrame):
#         data_to_adjust = data_to_adjust.to_pandas()

#     # Ensure fixed_time_horizons is a list
#     if not isinstance(fixed_time_horizons, list):
#         fixed_time_horizons = [fixed_time_horizons]

#     data = data_to_adjust.copy()

#     # Encode event types
#     event_map = {
#         "real_negatives": 0,
#         "real_positives": 1,
#         "real_competing": 2,
#         "real_censored": 0
#     }
#     data["event_code"] = data["reals"].map(event_map)

#     results = []

#     strata_values = data["strata"].unique()
#     for stratum in strata_values:
#         stratum_data = data.loc[data["strata"] == stratum]
#         n = len(stratum_data)

#         ajf_1 = AalenJohansenFitter()
#         ajf_1.fit(stratum_data["times"], stratum_data["event_code"], event_of_interest=1)

#         ajf_2 = AalenJohansenFitter()
#         ajf_2.fit(stratum_data["times"], stratum_data["event_code"], event_of_interest=2)

#         for t in fixed_time_horizons:
#             est_pos = ajf_1.predict(t)
#             est_comp = ajf_2.predict(t)
#             est_neg = 1 - est_pos - est_comp

#             for state, est in zip(
#                 ["real_negatives", "real_positives", "real_competing"],
#                 [est_neg, est_pos, est_comp]
#             ):
#                 results.append({
#                     "strata": stratum,
#                     "fixed_time_horizon": t,
#                     "reals": state,
#                     "n": n,
#                     "estimate": est,
#                     "reals_estimate": est * n
#                 })

#     result_df = pd.DataFrame(results)
#     result_df["strata"] = pd.Categorical(result_df["strata"])
#     return result_df


def extract_aj_estimate_polars(data_to_adjust, fixed_time_horizons):
    pass


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


def extract_crude_estimate(data_to_adjust: pd.DataFrame) -> pd.DataFrame:
    df = safe_pl_from_pandas(data_to_adjust)

    crude_estimate = df.group_by(["strata", "reals", "fixed_time_horizon"]).agg(
        pl.count().alias("reals_estimate")
    )

    unique_strata = df.select("strata").unique().to_series().to_list()
    unique_reals = df.select("reals").unique().to_series().to_list()
    unique_horizons = df.select("fixed_time_horizon").unique().to_series().to_list()

    all_combinations = pl.DataFrame(
        itertools.product(unique_strata, unique_reals, unique_horizons),
        schema=["strata", "reals", "fixed_time_horizon"],
    )

    final = all_combinations.join(
        crude_estimate, on=["strata", "reals", "fixed_time_horizon"], how="left"
    ).fill_null(0)

    return final.to_pandas()


# def extract_crude_estimate(data_to_adjust):
#     """
#     Computes the crude estimate by counting occurrences of 'reals' within
#     each combination of 'strata' and 'fixed_time_horizon'.

#     Args:
#         data_to_adjust (pd.DataFrame): Data containing 'strata', 'reals', and 'fixed_time_horizon'.

#     Returns:
#         pd.DataFrame: Aggregated counts with missing combinations filled with zero.
#     """

#     crude_estimate = (
#         data_to_adjust
#         .groupby(["strata", "reals", "fixed_time_horizon"], dropna=False)
#         .size()
#         .reset_index(name="reals_estimate")
#     )

#     unique_strata = data_to_adjust["strata"].unique()
#     unique_time_horizons = data_to_adjust["fixed_time_horizon"].unique()
#     unique_reals = data_to_adjust["reals"].unique()

#     all_combinations = pd.DataFrame(
#         list(itertools.product(unique_strata, unique_reals, unique_time_horizons)),
#         columns=["strata", "reals", "fixed_time_horizon"]
#     )

#     crude_estimate = (
#         all_combinations
#         .merge(crude_estimate, on=["strata", "reals", "fixed_time_horizon"], how="left")
#         .fillna({"reals_estimate": 0})
#     )

#     return crude_estimate


def add_cutoff_strata_polars(data: pl.DataFrame, by: float) -> pl.DataFrame:
    def transform_group(group: pl.DataFrame) -> pl.DataFrame:
        # Convert to NumPy for numeric ops
        probs = group["probs"].to_numpy()

        # --- Compute strata_probability_threshold ---
        breaks = create_breaks_values(probs, "probability_threshold", by)
        # strata_prob = np.digitize(probs, breaks, right=False) - 1
        # Clamp indices to avoid out-of-bounds error when accessing breaks[i+1]
        # strata_prob = np.clip(strata_prob, 0, len(breaks) - 2)
        # strata_prob_labels = [
        #     f"({breaks[i]:.3f}, {breaks[i+1]:.3f}]" for i in strata_prob
        # ]

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
        )

        # --- Compute strata_ppcr as quantiles on -probs ---
        try:
            q = int(1 / by)
            quantile_edges = np.quantile(-probs, np.linspace(0, 1, q))
            strata_ppcr = np.digitize(-probs, quantile_edges, right=False)
            strata_ppcr = (strata_ppcr / (1 / by)).astype(str)
        except ValueError:
            strata_ppcr = np.array(["1"] * len(probs))  # fallback for small group

        return group.with_columns(
            [
                pl.Series("strata_probability_threshold", strata_prob_labels),
                pl.Series("strata_ppcr", strata_ppcr),
            ]
        )

    # Apply per-group transformation
    grouped = data.partition_by("reference_group", as_dict=True)
    transformed_groups = [transform_group(group) for group in grouped.values()]
    return pl.concat(transformed_groups)


def add_cutoff_strata(data, by):
    result = data.copy()

    grouped = result.groupby("reference_group")

    def transform_group(group):
        group["strata_probability_threshold"] = pd.cut(
            group["probs"],
            bins=create_breaks_values(group["probs"], "probability_threshold", by),
            include_lowest=True,
        )

        group["strata_ppcr"] = (
            pd.qcut(-group["probs"], q=int(1 / by), labels=False, duplicates="drop") + 1
        )

        group["strata_ppcr"] = (group["strata_ppcr"] / (1 / by)).astype(str)

        return group

    result = grouped.apply(transform_group)

    result = result.reset_index(drop=True)

    return result


def create_breaks_values(_, stratified_by, by):
    # Dummy implementation; replace with your actual logic
    return np.arange(by, 1 + by, by)  # e.g., [0.1, 0.2, ..., 1.0]


def create_strata_combinations_polars(stratified_by: str, by: float) -> pl.DataFrame:
    if stratified_by == "probability_threshold":
        breaks = create_breaks_values(None, "probability_threshold", by)

        upper_bound = breaks[1:]  # breaks
        lower_bound = breaks[:-1]  # np.roll(upper_bound, 1)
        # lower_bound[0] = 0.0
        mid_point = upper_bound - by / 2
        include_lower_bound = lower_bound > -0.1
        include_upper_bound = upper_bound == 1.0  # upper_bound != 0.0
        chosen_cutoff = upper_bound
        strata = format_strata_column(
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            include_lower_bound=include_lower_bound,
            include_upper_bound=include_upper_bound,
            decimals=2,
        )

    elif stratified_by == "ppcr":
        strata_mid = create_breaks_values(None, "probability_threshold", by)[1:]
        lower_bound = strata_mid - by
        upper_bound = strata_mid + by
        mid_point = upper_bound - by
        include_lower_bound = np.ones_like(strata_mid, dtype=bool)
        include_upper_bound = np.zeros_like(strata_mid, dtype=bool)
        chosen_cutoff = strata_mid
        strata = np.round(mid_point, 3).astype(str)
    else:
        raise ValueError(f"Unsupported stratified_by: {stratified_by}")

    return pl.DataFrame(
        {
            "strata": pl.Series(strata),
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "mid_point": mid_point,
            "include_lower_bound": include_lower_bound,
            "include_upper_bound": include_upper_bound,
            "chosen_cutoff": chosen_cutoff,
            "stratified_by": [stratified_by] * len(strata),
        }
    )


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


def create_strata_combinations(stratified_by, by):
    if stratified_by == "probability_threshold":
        upper_bound = create_breaks_values(None, "probability_threshold", by)
        lower_bound = np.roll(upper_bound, 1)
        lower_bound[0] = 0
        mid_point = upper_bound - by / 2
        include_lower_bound = lower_bound == 0
        include_upper_bound = upper_bound != 0
        strata = [
            f"{'[' if include_lower else '('}{lower}, {upper}{']' if include_upper else ')'}"
            for include_lower, lower, upper, include_upper in zip(
                include_lower_bound, lower_bound, upper_bound, include_upper_bound
            )
        ]
        chosen_cutoff = upper_bound
    elif stratified_by == "ppcr":
        strata = create_breaks_values(None, "probability_threshold", by)[1:]
        lower_bound = strata - by
        upper_bound = strata + by
        mid_point = upper_bound - by / 2
        include_lower_bound = np.ones_like(strata, dtype=bool)
        include_upper_bound = np.zeros_like(strata, dtype=bool)
        chosen_cutoff = strata
    return pd.DataFrame(
        {
            "strata": strata,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "mid_point": mid_point,
            "include_lower_bound": include_lower_bound,
            "include_upper_bound": include_upper_bound,
            "chosen_cutoff": chosen_cutoff,
            "stratified_by": stratified_by,
        }
    )


def create_breaks_values_polars(probs_vec, stratified_by, by):
    # Ensure probs_vec is a NumPy array (in case it's a Polars Series)
    if hasattr(probs_vec, "to_numpy"):
        probs_vec = probs_vec.to_numpy()

    if stratified_by != "probability_threshold":
        # Quantile-based bin edges (descending)
        breaks = np.quantile(probs_vec, np.linspace(1, 0, int(1 / by) + 1))
    else:
        # Fixed-width bin edges (ascending)
        decimal_places = len(str(by).split(".")[-1])
        breaks = np.round(np.arange(0, 1 + by, by), decimals=decimal_places)

    return breaks


def create_breaks_values(probs_vec, stratified_by, by):
    if stratified_by != "probability_threshold":
        breaks = np.quantile(probs_vec, np.linspace(1, 0, int(1 / by) + 1))
    else:
        breaks = np.round(
            np.arange(0, 1 + by, by), decimals=len(str(by).split(".")[-1])
        )
    return breaks


def create_aj_data_combinations_polars(
    reference_groups, fixed_time_horizons, stratified_by, by
):
    # Create strata combinations using Polars
    strata_combinations_list = [
        create_strata_combinations_polars(x, by) for x in stratified_by
    ]
    strata_combinations = pl.concat(strata_combinations_list, how="vertical")

    strata_labels = strata_combinations["strata"]
    strata_enum = pl.Enum(strata_labels)

    strata_combinations = strata_combinations.with_columns(
        [pl.col("strata").cast(strata_enum)]
    )

    # Define values for Cartesian product
    reals_labels = [
        "real_negatives",
        "real_positives",
        "real_competing",
        "real_censored",
    ]
    reals_enum = pl.Enum(reals_labels)
    df_reals = pl.DataFrame({"reals_labels": pl.Series(reals_labels, dtype=reals_enum)})
    df_reference_groups = pl.DataFrame(
        {
            "reference_group": pl.Series(
                reference_groups, dtype=pl.Enum(reference_groups)
            )
        }
    )

    censoring_assumptions_labels = ["excluded", "adjusted"]
    censoring_assumptions_enum = pl.Enum(censoring_assumptions_labels)
    df_censoring_assumptions = pl.DataFrame(
        {
            "censoring_assumption": pl.Series(
                censoring_assumptions_labels, dtype=censoring_assumptions_enum
            )
        }
    )

    competing_assumptions_labels = [
        "excluded",
        "adjusted_as_negative",
        "adjusted_as_censored",
    ]
    competing_assumptions_enum = pl.Enum(competing_assumptions_labels)
    df_competing_assumptions = pl.DataFrame(
        {
            "competing_assumption": pl.Series(
                competing_assumptions_labels, dtype=competing_assumptions_enum
            )
        }
    )

    # Create all combinations
    combinations = list(
        itertools.product(
            # reference_groups,
            fixed_time_horizons,
            # censoring_assumptions,
            # competing_assumptions
        )
    )

    df_combinations = pl.DataFrame(
        combinations,
        schema=[
            # "reference_group",               # str
            "fixed_time_horizon",  # cast to Float64
            # "censoring_assumption",         # str
            # "competing_assumption"          # str
        ],
    ).with_columns(
        [
            pl.col("fixed_time_horizon").cast(pl.Int64),
            # pl.col("censoring_assumption").cast(pl.String),
            # pl.col("competing_assumption").cast(pl.String),
            # pl.col("reference_group").cast(pl.String)
        ]
    )

    # Cross join (cartesian product) with strata_combinations
    return (
        df_reference_groups.join(df_combinations, how="cross")
        .join(df_censoring_assumptions, how="cross")
        .join(df_competing_assumptions, how="cross")
        .join(strata_combinations, how="cross")
        .join(df_reals, how="cross")
    )


def create_aj_data_combinations(
    reference_groups, fixed_time_horizons, stratified_by, by
):
    strata_combinations = pd.concat(
        [create_strata_combinations(x, by) for x in stratified_by], ignore_index=True
    )

    reals = pd.Categorical(
        ["real_negatives", "real_positives", "real_competing", "real_censored"],
        categories=[
            "real_negatives",
            "real_positives",
            "real_competing",
            "real_censored",
        ],
        ordered=True,
    )

    censoring_assumptions = ["excluded", "adjusted"]
    competing_assumptions = ["excluded", "adjusted_as_negative", "adjusted_as_censored"]

    combinations = list(
        itertools.product(
            reference_groups,
            fixed_time_horizons,
            reals,
            censoring_assumptions,
            competing_assumptions,
        )
    )

    df_combinations = pd.DataFrame(
        combinations,
        columns=[
            "reference_group",
            "fixed_time_horizon",
            "reals",
            "censoring_assumption",
            "competing_assumption",
        ],
    )

    return df_combinations.merge(strata_combinations, how="cross")


def pivot_longer_strata_polars(data: pl.DataFrame) -> pl.DataFrame:
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

    # Remove "strata_" prefix from the 'stratified_by' column
    data_long = data_long.with_columns(
        pl.col("stratified_by").str.replace("^strata_", "")
    )

    return data_long


def pivot_longer_strata(data):
    data = data.copy()  # Ensure we are not modifying the original DataFrame

    # Melt the DataFrame, converting multiple 'strata_*' columns into long format
    data_long = data.melt(
        id_vars=[
            col for col in data.columns if not col.startswith("strata_")
        ],  # Keep all non-strata columns
        value_vars=[
            col for col in data.columns if col.startswith("strata_")
        ],  # Melt only strata columns
        var_name="stratified_by",
        value_name="strata",
    )

    # Remove "strata_" prefix from stratified_by column (equivalent to `names_prefix = "strata_"` in R)
    data_long["stratified_by"] = data_long["stratified_by"].str.replace("strata_", "")

    return data_long


def update_administrative_censoring(data_to_adjust: pd.DataFrame) -> pd.DataFrame:
    data_to_adjust = data_to_adjust.copy()
    data_to_adjust["reals"] = data_to_adjust["reals"].astype(str)

    pl_data = safe_pl_from_pandas(data_to_adjust)

    # Define logic in Python and map it row-wise (this avoids any column reference issues)
    def adjust(row):
        t = row["times"]
        h = row["fixed_time_horizon"]
        r = row["reals"]
        if t > h and r == "real_positives":
            return "real_negatives"
        if t < h and r == "real_negatives":
            return "real_censored"
        return r

    pl_data = pl_data.with_columns(
        [
            pl.struct(["times", "fixed_time_horizon", "reals"])
            .map_elements(adjust)
            .alias("reals")
        ]
    )

    return pl_data.to_pandas()


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


def extract_aj_estimate_for_strata(
  data_to_adjust, 
  fixed_time_horizons_for_aj):
  
  ajf_primary = AalenJohansenFitter()
  ajf_competing = AalenJohansenFitter()

  ajf_primary.fit(
    data_to_adjust['times'].to_pandas(),
    data_to_adjust['reals'].to_pandas(),
    event_of_interest=1
  )

  ajf_competing.fit(
    data_to_adjust['times'].to_pandas(),
    data_to_adjust['reals'].to_pandas(),
    event_of_interest=2
  )

  n = data_to_adjust.height

  real_positives_est = ajf_primary.predict(fixed_time_horizons_for_aj)
  real_competing_est = ajf_competing.predict(fixed_time_horizons_for_aj)
  real_negatives_est = 1 - real_positives_est - real_competing_est

  return pl.DataFrame({
    "strata": data_to_adjust['strata'][0],
    "fixed_time_horizon": fixed_time_horizons_for_aj,
    "real_negatives_est": real_negatives_est*n,
    "real_positives_est": real_positives_est*n,
    "real_competing_est": real_competing_est*n
  })


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


def extract_aj_estimate_for_strata(
  data_to_adjust, 
  fixed_time_horizons_for_aj):
  
  ajf_primary = AalenJohansenFitter()
  ajf_competing = AalenJohansenFitter()

  ajf_primary.fit(
    data_to_adjust['times'].to_pandas(),
    data_to_adjust['reals'].to_pandas(),
    event_of_interest=1
  )

  ajf_competing.fit(
    data_to_adjust['times'].to_pandas(),
    data_to_adjust['reals'].to_pandas(),
    event_of_interest=2
  )

  n = data_to_adjust.height

  real_positives_est = ajf_primary.predict(fixed_time_horizons_for_aj)
  real_competing_est = ajf_competing.predict(fixed_time_horizons_for_aj)
  real_negatives_est = 1 - real_positives_est - real_competing_est

  return pl.DataFrame({
    "strata": data_to_adjust['strata'][0],
    "fixed_time_horizon": fixed_time_horizons_for_aj,
    "real_negatives_est": real_negatives_est*n,
    "real_positives_est": real_positives_est*n,
    "real_competing_est": real_competing_est*n
  })

def create_adjusted_data_list_polars(
    list_data_to_adjust, fixed_time_horizons, assumption_sets
):
    adjusted_data_list = []

    for reference_group, group_data_polars in list_data_to_adjust.items():
        for assumptions in assumption_sets:
            adjusted_data = extract_aj_estimate_by_assumptions_polars(
                group_data_polars,
                fixed_time_horizons=fixed_time_horizons,
                censoring_assumption=assumptions["censored"],
                competing_assumption=assumptions["competing"],
            )

            adjusted_data = adjusted_data.with_columns(
                pl.lit(reference_group).alias("reference_group")
            )
            adjusted_data_list.append(adjusted_data)

    return adjusted_data_list


def create_adjusted_data_list(
    list_data_to_adjust, fixed_time_horizons, assumption_sets
):
    adjusted_data_list = []
    for reference_group, group_data in list_data_to_adjust.items():
        group_data = ensure_arrow_safe(group_data)

        for assumptions in assumption_sets:
            adjusted_data = extract_aj_estimate_by_assumptions(
                group_data,
                fixed_time_horizons=fixed_time_horizons,
                censoring_assumption=assumptions["censored"],
                competing_assumption=assumptions["competing"],
            )
            adjusted_data["reference_group"] = reference_group
            adjusted_data_list.append(adjusted_data)
    return adjusted_data_list


# def assign_and_explode(data, fixed_time_horizons):
#     data = data.copy()
#     data["fixed_time_horizon"] = [fixed_time_horizons] * len(data)
#     data = data.explode("fixed_time_horizon")
#     return data


def assign_and_explode(data: pd.DataFrame, fixed_time_horizons) -> pd.DataFrame:
    # Ensure list type
    if not isinstance(fixed_time_horizons, list):
        fixed_time_horizons = [fixed_time_horizons]

    # Convert safely to Polars
    df = safe_pl_from_pandas(data)

    # Add the repeated list to each row, then explode
    df = df.with_columns(
        pl.Series("fixed_time_horizon", [fixed_time_horizons] * df.height)
    ).explode("fixed_time_horizon")

    return df.to_pandas()


def assign_and_explode_polars(
    data: pl.DataFrame, fixed_time_horizons: list[float]
) -> pl.DataFrame:
    return (
        data.with_columns(pl.lit(fixed_time_horizons).alias("fixed_time_horizon"))
        .explode("fixed_time_horizon")
        .with_columns(pl.col("fixed_time_horizon").cast(pl.Float64))
    )


# def extract_aj_estimate_by_assumptions_polars(
#     data_to_adjust: pl.DataFrame,
#     fixed_time_horizons: list[float],
#     censoring_assumption: str = "excluded",
#     competing_assumption: str = "excluded"
# ) -> pl.DataFrame:

#     if censoring_assumption == "excluded" and competing_assumption == "excluded":
#         print("🪵 Step 1: Input columns:", data_to_adjust.columns)

#         aj = assign_and_explode_polars(data_to_adjust, fixed_time_horizons)
#         print("🪵 Step 2: After assign_and_explode columns:", aj.columns)
#         print("🪵 Schema:", aj.schema)
#         print("🪵 Sample rows:")
#         print(aj.head(3))

#         aj = update_administrative_censoring_polars(aj)
#         print("🪵 Step 3: After update_administrative_censoring:", aj.columns)

#         return aj.with_columns([
#             pl.lit(censoring_assumption).alias("censoring_assumption"),
#             pl.lit(competing_assumption).alias("competing_assumption")
#         ])


def extract_aj_estimate_by_assumptions_polars(
    data_to_adjust: pl.DataFrame,
    fixed_time_horizons: list[float],
    censoring_assumption="excluded",
    competing_assumption="excluded",
) -> pl.DataFrame:
    def to_pd(df):
        return df.to_pandas()

    def to_pl(df):
        return pl.from_pandas(df)

    if censoring_assumption == "excluded" and competing_assumption == "excluded":
        aj_estimate_data = (
            assign_and_explode_polars(data_to_adjust, fixed_time_horizons)
            .pipe(update_administrative_censoring_polars)
            .pipe(extract_crude_estimate_polars)
        )

        aj_estimate_data = aj_estimate_data.with_columns(
            pl.col("reals_estimate").cast(pl.Float64).alias("reals_estimate")
        )

        aj_estimate_data = aj_estimate_data.with_columns(
            pl.col("strata").cast(pl.Categorical).alias("strata")
        )

        aj_estimate_data = aj_estimate_data.with_columns(
            pl.col("fixed_time_horizon").cast(pl.Int64).alias("fixed_time_horizon")
        )

    if censoring_assumption == "adjusted" and competing_assumption == "excluded":
        exploded = assign_and_explode_polars(data_to_adjust, fixed_time_horizons)
        exploded = update_administrative_censoring_polars(exploded)

        # Separate "real_competing" for crude estimation
        real_competing_data = exploded.filter(
            pl.col("reals_labels") == "real_competing"
        )
        non_competing_data = exploded.filter(pl.col("reals_labels") != "real_competing")

        # Crude estimate for "real_competing" using Polars
        aj_estimate_competing = extract_crude_estimate_polars(real_competing_data)

        aj_estimate_competing = aj_estimate_competing.with_columns(
            pl.col("strata").cast(pl.Categorical).alias("strata")
        )

        aj_estimate_competing = aj_estimate_competing.with_columns(
            pl.col("fixed_time_horizon").cast(pl.Int64).alias("fixed_time_horizon")
        )

        aj_estimate_competing = aj_estimate_competing.with_columns(
            pl.col("reals_estimate").cast(pl.Float64).alias("reals_estimate")
        )

        # Aalen-Johansen estimate for non-competing using Lifelines (pandas)
        aj_estimate_adjusted_list = [
            extract_aj_estimate(
                to_pd(non_competing_data.filter(pl.col("fixed_time_horizon") == h)),
                fixed_time_horizons=[h],
            )
            for h in fixed_time_horizons
        ]

        # Combine results
        aj_estimate_adjusted = to_pl(
            pd.concat(aj_estimate_adjusted_list, ignore_index=True)
        )

        reals_labels = [
            "real_negatives",
            "real_positives",
            "real_competing",
            "real_censored",
        ]
        reals_enum = pl.Enum(reals_labels)

        aj_estimate_adjusted = aj_estimate_adjusted.with_columns(
            pl.col("reals").cast(reals_enum).alias("reals")
        )

        aj_estimate_data = pl.concat([aj_estimate_competing, aj_estimate_adjusted])

    return aj_estimate_data.with_columns(
        [
            pl.lit(censoring_assumption).alias("censoring_assumption"),
            pl.lit(competing_assumption).alias("competing_assumption"),
        ]
    )


def extract_aj_estimate_by_assumptions(
    data_to_adjust: pd.DataFrame,
    fixed_time_horizons,
    censoring_assumption="excluded",
    competing_assumption="excluded",
) -> pd.DataFrame:
    def to_pl(df):
        return safe_pl_from_pandas(df)

    def to_pd(df):
        return df.to_pandas()

    if censoring_assumption == "excluded" and competing_assumption == "excluded":
        aj_estimate_data = (
            assign_and_explode(data_to_adjust, fixed_time_horizons)
            .pipe(update_administrative_censoring)
            .pipe(extract_crude_estimate)
        )

    elif (
        censoring_assumption == "excluded"
        and competing_assumption == "adjusted_as_negative"
    ):
        exploded = assign_and_explode(data_to_adjust, fixed_time_horizons)
        exploded = update_administrative_censoring(exploded)
        pl_exploded = to_pl(exploded)

        aj_estimate_data_excluded = extract_crude_estimate(
            to_pd(pl_exploded.filter(pl.col("reals") == "real_censored"))
        )

        aj_estimate_data_adjusted = pd.concat(
            [
                extract_aj_estimate(
                    data_to_adjust.assign(fixed_time_horizon=h).query(
                        "reals != 'real_censored' and fixed_time_horizon == @h"
                    ),
                    fixed_time_horizons=h,
                )
                for h in fixed_time_horizons
            ],
            ignore_index=True,
        )

        aj_estimate_data = pd.concat(
            [aj_estimate_data_excluded, aj_estimate_data_adjusted], ignore_index=True
        )

    elif censoring_assumption == "adjusted" and competing_assumption == "excluded":
        exploded = assign_and_explode(data_to_adjust, fixed_time_horizons)
        exploded = update_administrative_censoring(exploded)
        pl_exploded = to_pl(exploded)

        aj_estimate_data_excluded = extract_crude_estimate(
            to_pd(pl_exploded.filter(pl.col("reals") == "real_competing"))
        )

        aj_estimate_data_adjusted = pd.concat(
            [
                extract_aj_estimate(
                    data_to_adjust.assign(fixed_time_horizon=h).query(
                        "reals != 'real_competing' and fixed_time_horizon == @h"
                    ),
                    fixed_time_horizons=h,
                )
                for h in fixed_time_horizons
            ],
            ignore_index=True,
        )

        aj_estimate_data = pd.concat(
            [aj_estimate_data_excluded, aj_estimate_data_adjusted], ignore_index=True
        )

    elif (
        censoring_assumption == "adjusted"
        and competing_assumption == "adjusted_as_negative"
    ):
        aj_estimate_data = extract_aj_estimate(data_to_adjust, fixed_time_horizons)

    elif (
        censoring_assumption == "adjusted"
        and competing_assumption == "adjusted_as_censored"
    ):
        df = data_to_adjust.copy()
        df["reals"] = df["reals"].replace({"real_competing": "real_negatives"})
        aj_estimate_data = extract_aj_estimate(df, fixed_time_horizons)

    return aj_estimate_data.assign(
        censoring_assumption=censoring_assumption,
        competing_assumption=competing_assumption,
    )


# def extract_aj_estimate_by_assumptions(data_to_adjust, fixed_time_horizons,
#                                        censoring_assumption="excluded",
#                                        competing_assumption="excluded"):


#     print("data_to_adjust")
#     print(data_to_adjust)

#     print("data_to_adjust.dtypes")
#     print(data_to_adjust.dtypes)

#     data_to_adjust = ensure_arrow_safe(data_to_adjust)

#     print("data_to_adjust after ensure no categorical")
#     print(data_to_adjust)

#     print("data_to_adjust.dtypes after ensure no categorical")
#     print(data_to_adjust.dtypes)


#     # def assign_and_explode(data):
#     #     return (
#     #         data.assign(fixed_time_horizon=[fixed_time_horizons] * len(data))
#     #           .explode("fixed_time_horizon")
#     #     )
#     # def assign_and_explode(data):
#     #     try:
#     #         import polars as pl
#     #     except ImportError as e:
#     #         raise ImportError("Polars is required for assign_and_explode. Please install it with 'pip install polars'.") from e

#     #     pl_data = pl.from_pandas(data)
#     #     pl_data = pl_data.with_columns(
#     #         pl.Series("fixed_time_horizon", [fixed_time_horizons] * pl_data.height)
#     #     ).explode("fixed_time_horizon")
#     #     return pl_data.to_pandas()

#     if censoring_assumption == "excluded" and competing_assumption == "excluded":

#         aj_estimate_data = (
#             assign_and_explode(data_to_adjust, fixed_time_horizons=fixed_time_horizons)
#             .pipe(update_administrative_censoring)
#             .pipe(extract_crude_estimate)
#         )

#     elif censoring_assumption == "excluded" and competing_assumption == "adjusted_as_negative":
#         exploded_data = assign_and_explode(data_to_adjust, fixed_time_horizons=fixed_time_horizons).pipe(update_administrative_censoring)

#         aj_estimate_data_excluded = (
#             exploded_data
#             .pipe(update_administrative_censoring)
#             .query("reals == 'real_censored'")
#             .pipe(extract_crude_estimate)
#         )

#         aj_estimate_data_adjusted = (
#             pd.concat([
#                 extract_aj_estimate(
#                     data_to_adjust
#                     .assign(fixed_time_horizon=h)
#                     .query("reals != 'real_censored' and fixed_time_horizon == @h"),
#                     fixed_time_horizons=h
#                 )
#                 for h in fixed_time_horizons
#             ])
#             .reset_index(drop=True)
#         )

#         aj_estimate_data = pd.concat([aj_estimate_data_excluded, aj_estimate_data_adjusted], ignore_index=True)

#     elif censoring_assumption == "adjusted" and competing_assumption == "excluded":
#         exploded_data = assign_and_explode(data_to_adjust, fixed_time_horizons=fixed_time_horizons).pipe(update_administrative_censoring)

#         aj_estimate_data_excluded = (
#             exploded_data
#             .query("reals == 'real_competing'")
#             .pipe(extract_crude_estimate)
#         )

#         aj_estimate_data_adjusted = (
#             pd.concat([
#                 extract_aj_estimate(
#                     data_to_adjust
#                     .assign(fixed_time_horizon=h)
#                     .query("reals != 'real_competing' and fixed_time_horizon == @h"),
#                     fixed_time_horizons=h
#                 )
#                 for h in fixed_time_horizons
#             ])
#             .reset_index(drop=True)
#         )

#         aj_estimate_data = pd.concat([aj_estimate_data_excluded, aj_estimate_data_adjusted], ignore_index=True)

#     elif censoring_assumption == "adjusted" and competing_assumption == "adjusted_as_negative":
#         aj_estimate_data = extract_aj_estimate(data_to_adjust, fixed_time_horizons=fixed_time_horizons)

#     elif censoring_assumption == "adjusted" and competing_assumption == "adjusted_as_censored":
#         aj_estimate_data = extract_aj_estimate(
#             data_to_adjust.assign(reals=data_to_adjust["reals"].replace({"real_competing": "real_negatives"})),
#             fixed_time_horizons=fixed_time_horizons
#         )

#     return aj_estimate_data.assign(
#         censoring_assumption=censoring_assumption,
#         competing_assumption=competing_assumption
#     )


def create_list_data_to_adjust_polars(
    probs_dict, reals_dict, times_dict, stratified_by, by
):
    reference_groups = list(probs_dict.keys())
    num_reals = len(reals_dict)

    # Flatten and ensure list format
    data_to_adjust = pl.DataFrame(
        {
            "reference_group": sum(
                [[group] * num_reals for group in reference_groups], []
            ),
            "probs": sum(
                [probs_dict[group].tolist() for group in reference_groups], []
            ),
            "reals": list(reals_dict) * len(reference_groups),
            "times": list(times_dict) * len(reference_groups),
        }
    )

    # Apply strata
    data_to_adjust = add_cutoff_strata_polars(data_to_adjust, by=by)
    data_to_adjust = pivot_longer_strata_polars(data_to_adjust)

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


def create_list_data_to_adjust(probs_dict, reals_dict, times_dict, stratified_by, by):
    reference_groups = list(probs_dict.keys())
    data_to_adjust = pd.DataFrame(
        {
            "reference_group": np.repeat(reference_groups, len(reals_dict)),
            "probs": np.concatenate([probs_dict[group] for group in reference_groups]),
            "reals": np.tile(reals_dict, len(reference_groups)),
            "times": np.tile(times_dict, len(reference_groups)),
        }
    )
    data_to_adjust = add_cutoff_strata(data_to_adjust, by=by)
    data_to_adjust = pivot_longer_strata(data_to_adjust)
    data_to_adjust["reals"] = data_to_adjust["reals"].replace(
        {0: "real_negatives", 2: "real_competing", 1: "real_positives"}
    )
    data_to_adjust["reals"] = pd.Categorical(
        data_to_adjust["reals"],
        categories=["real_negatives", "real_competing", "real_positives"],
        ordered=True,
    )
    list_data_to_adjust = {k: v for k, v in data_to_adjust.groupby("reference_group")}
    return list_data_to_adjust


def safe_pl_from_pandas(df: pd.DataFrame) -> pl.DataFrame:
    df = df.copy()
    for col in df.select_dtypes(include="category").columns:
        df[col] = df[col].astype(str)
    for col in df.columns:
        if df[col].dtype == "object":
            try:
                if any(
                    isinstance(val, pd._libs.interval.Interval)
                    for val in df[col].dropna()
                ):
                    df[col] = df[col].astype(str)
            except Exception:
                df[col] = df[col].astype(str)
    return pl.from_pandas(df)


def ensure_no_categorical(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.select_dtypes(include="category").columns:
        df[col] = df[col].astype(str)
    return df


def ensure_arrow_safe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Convert all category columns to string
    for col in df.select_dtypes(include="category").columns:
        df[col] = df[col].astype(str)

    # Convert Interval and other Arrow-unsafe objects to string
    for col in df.columns:
        if df[col].dtype == "object":
            try:
                # Try to catch Interval or any other problematic type
                if any(
                    isinstance(val, pd._libs.interval.Interval)
                    for val in df[col].dropna()
                ):
                    df[col] = df[col].astype(str)
            except Exception:
                df[col] = df[col].astype(
                    str
                )  # fallback: convert whole column to string

    return df

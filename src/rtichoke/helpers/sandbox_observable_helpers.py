import pandas as pd
import numpy as np
import itertools

def extract_aj_estimate(data_to_adjust, fixed_time_horizons):
    """
    Python implementation of the R extract_aj_estimate function for Aalen-Johansen estimation.
    
    Parameters:
    data_to_adjust (pd.DataFrame): DataFrame containing survival data
    fixed_time_horizons (list or float): Time points at which to evaluate the survival
    
    Returns:
    pd.DataFrame: DataFrame with Aalen-Johansen estimates
    """
    import numpy as np

    # print(f"data_to_adjust: {data_to_adjust}")
    
    # Ensure 'strata' column exists
    if 'strata' not in data_to_adjust.columns:
        data_to_adjust['strata'] = 'default'

    # Ensure fixed_time_horizons is a list
    if not isinstance(fixed_time_horizons, list):
        fixed_time_horizons = [fixed_time_horizons]
    
    # Create a categorical version of reals for stratification
    data_to_adjust = data_to_adjust.copy()
    data_to_adjust['reals_cat'] = pd.Categorical(
        data_to_adjust['reals'], 
        categories=["real_negatives", "real_positives", "real_competing", "real_censored"],
        ordered=True
    )
    
    # Get unique strata values
    strata_values = data_to_adjust['strata'].unique()
    
    # Initialize result dataframes
    results = []
    
    # For each stratum, fit Aalen-Johansen model
    for stratum in strata_values:
        # Filter data for current stratum
        stratum_data = data_to_adjust[data_to_adjust['strata'] == stratum].copy()
        
        # Initialize Aalen-Johansen fitter
        ajf = AalenJohansenFitter()
        
        # Convert reals to numeric for lifelines
        # In lifelines: 0=censored, 1=event of interest, 2=competing event
        event_map = {
            "real_negatives": 0,  # Treat as censored
            "real_positives": 1,  # Event of interest
            "real_competing": 2,  # Competing risk
            "real_censored": 0    # Censored
        }
        
        stratum_data['event_code'] = stratum_data['reals'].map(event_map)
        
        # Fit the model
        ajf.fit(
            stratum_data['times'], 
            stratum_data['event_code'], 
            event_of_interest=1
        )
        
        # Calculate cumulative incidence at fixed time horizons
        for t in fixed_time_horizons:
            # Get cumulative incidence at time t
            # Handle case where t is beyond the data
            try:
                ci_at_t = ajf.cumulative_density_at_times(t)[0]
            except IndexError:
                ci_at_t = ajf.cumulative_density_.iloc[-1] if len(ajf.cumulative_density_) > 0 else 0
            
            # Calculate counts for each state
            n = len(stratum_data)
            
            # For real_positives: use the cumulative incidence
            real_positives_est = ci_at_t
            
            # For real_negatives: use 1 - cumulative incidence
            real_negatives_est = 1 - ci_at_t
            
            # For real_competing: would be obtained from a competing risks model
            # Here we'll use the count of competing events before time t
            real_competing_count = stratum_data[(stratum_data['event_code'] == 2) & 
                                               (stratum_data['times'] <= t)].shape[0]
            real_competing_est = real_competing_count / n if n > 0 else 0
            
            # For real_censored: would be the proportion of censored before time t
            real_censored_count = stratum_data[(stratum_data['event_code'] == 0) & 
                                              (stratum_data['times'] <= t)].shape[0]
            real_censored_est = real_censored_count / n if n > 0 else 0
            
            # Adjust estimates to ensure they sum to 1
            total_est = real_positives_est + real_negatives_est + real_competing_est + real_censored_est
            if total_est > 0:
                real_positives_est /= total_est
                real_negatives_est /= total_est
                real_competing_est /= total_est
                real_censored_est /= total_est
            
            # Create entries for each real state
            states = ["real_negatives", "real_positives", "real_competing", "real_censored"]
            estimates = [real_negatives_est, real_positives_est, real_competing_est, real_censored_est]
            
            for state, estimate in zip(states, estimates):
                results.append({
                    'strata': stratum,
                    'fixed_time_horizon': t,
                    'reals': state,
                    'n': n,
                    'estimate': estimate,
                    'reals_estimate': estimate * n
                })
    
    # Convert to DataFrame
    result_df = pd.DataFrame(results)
    
    # Convert strata to categorical if needed
    result_df['strata'] = pd.Categorical(result_df['strata'])
    
    return result_df



def extract_crude_estimate(data_to_adjust):
    """
    Computes the crude estimate by counting occurrences of 'reals' within 
    each combination of 'strata' and 'fixed_time_horizon'.
    
    Args:
        data_to_adjust (pd.DataFrame): Data containing 'strata', 'reals', and 'fixed_time_horizon'.
        
    Returns:
        pd.DataFrame: Aggregated counts with missing combinations filled with zero.
    """
    # Group by strata, reals, and fixed_time_horizon, then count occurrences
    
    # print('data_to_adjust')
    # print(data_to_adjust)
    
    crude_estimate = (
        data_to_adjust
        .groupby(["strata", "reals", "fixed_time_horizon"], dropna=False)
        .size()
        .reset_index(name="reals_estimate")
    )

    unique_strata = data_to_adjust["strata"].unique()
    unique_time_horizons = data_to_adjust["fixed_time_horizon"].unique()
    unique_reals = data_to_adjust["reals"].unique()

    # return crude_estimate

    # Create all possible combinations to ensure completeness
    all_combinations = pd.DataFrame(
        list(itertools.product(unique_strata, unique_reals, unique_time_horizons)),
        columns=["strata", "reals", "fixed_time_horizon"]
    )


    # Ensure all possible combinations are present and fill missing values with 0
    crude_estimate = (
        all_combinations
        .merge(crude_estimate, on=["strata", "reals", "fixed_time_horizon"], how="left")
        .fillna({"reals_estimate": 0})
    )

    return crude_estimate

def add_cutoff_strata(data, by):
    data["strata_probability_threshold"] = pd.cut(
        data["probs"],
        bins=create_breaks_values(data["probs"], "probability_threshold", by),
        include_lowest=True
    )
    data["strata_ppcr"] = (pd.qcut(-data["probs"], q=int(1/by), labels=False) + 1) / (1 / by)
    data["strata_ppcr"] = data["strata_ppcr"].astype(str)
    return data



def create_strata_combinations(stratified_by, by):
    if stratified_by == "probability_threshold":
        upper_bound = create_breaks_values(None, "probability_threshold", by)
        lower_bound = np.roll(upper_bound, 1)
        lower_bound[0] = 0
        mid_point = upper_bound - by / 2
        include_lower_bound = lower_bound == 0
        include_upper_bound = upper_bound != 0
        strata = [f"{'[' if lb else '('}{l}, {u}{']' if ub else ')'}" for lb, l, u, ub in zip(include_lower_bound, lower_bound, upper_bound, include_upper_bound)]
        chosen_cutoff = upper_bound
    elif stratified_by == "ppcr":
        strata = create_breaks_values(None, "probability_threshold", by)[1:]
        lower_bound = strata - by
        upper_bound = strata + by
        mid_point = upper_bound - by / 2
        include_lower_bound = np.ones_like(strata, dtype=bool)
        include_upper_bound = np.zeros_like(strata, dtype=bool)
        chosen_cutoff = strata
    return pd.DataFrame({
        "strata": strata,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "mid_point": mid_point,
        "include_lower_bound": include_lower_bound,
        "include_upper_bound": include_upper_bound,
        "chosen_cutoff": chosen_cutoff,
        "stratified_by": stratified_by
    })



def create_breaks_values(probs_vec, stratified_by, by):
    if stratified_by != "probability_threshold":
        breaks = np.quantile(probs_vec, np.linspace(1, 0, int(1/by) + 1))
    else:
        breaks = np.round(np.arange(0, 1 + by, by), decimals=len(str(by).split(".")[-1]))
    return breaks


def create_aj_data_combinations(reference_groups, fixed_time_horizons, stratified_by, by):
    # Create strata combinations
    strata_combinations = pd.concat(
        [create_strata_combinations(x, by) for x in stratified_by], ignore_index=True
    )
    
    reals = pd.Categorical(
        ["real_negatives", "real_positives", "real_competing", "real_censored"],
        categories=["real_negatives", "real_positives", "real_competing", "real_censored"],
        ordered=True
    )
    
    censoring_assumptions = ["excluded", "adjusted"]
    competing_assumptions = ["excluded", "adjusted_as_negative", "adjusted_as_censored"]
    
    combinations = list(itertools.product(reference_groups, fixed_time_horizons, reals, censoring_assumptions, competing_assumptions))
    
    df_combinations = pd.DataFrame(combinations, columns=[
        "reference_group", "fixed_time_horizon", "reals", "censoring_assumption", "competing_assumption"
    ])
    
    return df_combinations.merge(strata_combinations, how="cross")


def pivot_longer_strata(data):
    data = data.copy()  # Ensure we are not modifying the original DataFrame
    
    # Melt the DataFrame, converting multiple 'strata_*' columns into long format
    data_long = data.melt(
        id_vars=[col for col in data.columns if not col.startswith("strata_")],  # Keep all non-strata columns
        value_vars=[col for col in data.columns if col.startswith("strata_")],  # Melt only strata columns
        var_name="stratified_by", 
        value_name="strata"
    )

    # Remove "strata_" prefix from stratified_by column (equivalent to `names_prefix = "strata_"` in R)
    data_long["stratified_by"] = data_long["stratified_by"].str.replace("strata_", "")

    return data_long

def update_administrative_censoring(data_to_adjust):
    data_to_adjust = data_to_adjust.copy()  # Ensure we're not modifying the original DataFrame

    data_to_adjust["reals"] = np.select(
        [
            (data_to_adjust["times"] > data_to_adjust["fixed_time_horizon"]) & (data_to_adjust["reals"] == "real_positives"),
            (data_to_adjust["times"] < data_to_adjust["fixed_time_horizon"]) & (data_to_adjust["reals"] == "real_negatives")
        ],
        [
            "real_negatives",  # If time is greater than horizon and was "real_positives", change to "real_negatives"
            "real_censored"    # If time is less than horizon and was "real_negatives", change to "real_censored"
        ],
        default=data_to_adjust["reals"]  # Keep the original value if no condition is met
    )
    return data_to_adjust


def extract_aj_estimate_by_assumptions(data_to_adjust, fixed_time_horizons, 
                                       censoring_assumption="excluded", 
                                       competing_assumption="excluded"):
    
    # print('censoring_assumption')
    # print(censoring_assumption)

    # print('competing assumption')
    # print(competing_assumption)

    
    if censoring_assumption == "excluded" and competing_assumption == "excluded":

        
        aj_estimate_data = (
            data_to_adjust
            .assign(fixed_time_horizon=lambda df: df.apply(lambda x: fixed_time_horizons, axis=1))
            .explode("fixed_time_horizon")
            .pipe(update_administrative_censoring)
            .pipe(extract_crude_estimate)
        )

        # print('aj_estimate-data')
        # print(aj_estimate_data)
    
    elif censoring_assumption == "excluded" and competing_assumption == "adjusted_as_negative":
        aj_estimate_data_excluded = (
            data_to_adjust
            .assign(fixed_time_horizon=lambda df: df.apply(lambda x: fixed_time_horizons, axis=1))
            .explode("fixed_time_horizon")
            .pipe(update_administrative_censoring)
            .query("reals == 'real_censored'")
            .pipe(extract_crude_estimate)
        )

        aj_estimate_data_adjusted = (
            pd.concat([
                extract_aj_estimate(
                    data_to_adjust
                    .assign(fixed_time_horizon=h)
                    .query("reals != 'real_censored' and fixed_time_horizon == @h"),
                    fixed_time_horizons=h
                )
                for h in fixed_time_horizons
            ])
            .reset_index(drop=True)
        )

        aj_estimate_data = pd.concat([aj_estimate_data_excluded, aj_estimate_data_adjusted], ignore_index=True)
    
    elif censoring_assumption == "adjusted" and competing_assumption == "excluded":
        aj_estimate_data_excluded = (
            data_to_adjust
            .assign(fixed_time_horizon=lambda df: df.apply(lambda x: fixed_time_horizons, axis=1))
            .explode("fixed_time_horizon")
            .pipe(update_administrative_censoring)
            .query("reals == 'real_competing'")
            .pipe(extract_crude_estimate)
        )

        aj_estimate_data_adjusted = (
            pd.concat([
                extract_aj_estimate(
                    data_to_adjust
                    .assign(fixed_time_horizon=h)
                    .query("reals != 'real_competing' and fixed_time_horizon == @h"),
                    fixed_time_horizons=h
                )
                for h in fixed_time_horizons
            ])
            .reset_index(drop=True)
        )

        aj_estimate_data = pd.concat([aj_estimate_data_excluded, aj_estimate_data_adjusted], ignore_index=True)
    
    elif censoring_assumption == "adjusted" and competing_assumption == "adjusted_as_negative":
        aj_estimate_data = extract_aj_estimate(data_to_adjust, fixed_time_horizons=fixed_time_horizons)
    
    elif censoring_assumption == "adjusted" and competing_assumption == "adjusted_as_censored":
        aj_estimate_data = extract_aj_estimate(
            data_to_adjust.assign(reals=data_to_adjust["reals"].replace({"real_competing": "real_negatives"})),
            fixed_time_horizons=fixed_time_horizons
        )

    return aj_estimate_data.assign(
        censoring_assumption=censoring_assumption,
        competing_assumption=competing_assumption
    )
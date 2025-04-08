from lifelines import CoxPHFitter
from lifelines import WeibullAFTFitter
import pandas as pd
import numpy as np
from rtichoke.helpers.sandbox_observable_helpers import add_cutoff_strata, create_aj_data_combinations, extract_aj_estimate_by_assumptions, pivot_longer_strata

df_time_to_cancer_dx = \
    pd.read_csv(
        "https://raw.githubusercontent.com/ddsjoberg/dca-tutorial/main/data/df_time_to_cancer_dx.csv"
    )


cox_model = CoxPHFitter()
thin_model = CoxPHFitter()
aft_model = WeibullAFTFitter()

cox_formula = 'age + famhistory + marker'
thin_formula = 'age + marker'
aft_formula = 'age + marker'

cox_model.fit(df_time_to_cancer_dx, duration_col='ttcancer', event_col='cancer', formula=cox_formula)
thin_model.fit(df_time_to_cancer_dx, duration_col='ttcancer', event_col='cancer', formula=thin_formula)
aft_model.fit(df_time_to_cancer_dx, duration_col='ttcancer', event_col='cancer', formula=aft_formula)



reals_mapping = {
    "censor": 0,
    "diagnosed with cancer": 1,
    "dead other causes": 2
}

df_time_to_cancer_dx['reals'] = df_time_to_cancer_dx['cancer_cr'].map(reals_mapping)

new_data = df_time_to_cancer_dx.copy()
new_data['ttcancer'] = 1.5

preds_aft = 1 - np.exp(-aft_model.predict_expectation(new_data))
pred_1_5 = 1 - np.exp(-cox_model.predict_expectation(new_data))
pred_thin = 1 - np.exp(-thin_model.predict_expectation(new_data))

probs_cox = {
    "thin": pred_thin,
    "full": pred_1_5,
    "aft": preds_aft
}



fixed_time_horizons = [1, 3, 5]
stratified_by = ["probability_threshold", "ppcr"]

aj_data_combinations = create_aj_data_combinations(list(probs_cox.keys()), fixed_time_horizons, stratified_by, 0.1)


# Create reference groups
data_to_adjust = pd.DataFrame({
    "reference_group": ["thin"] * len(probs_cox["thin"]) + ["aft"] * len(probs_cox["thin"]) + ["full"] * len(probs_cox["thin"]),
    "probs": np.concatenate([probs_cox["thin"], probs_cox["aft"], probs_cox["full"]]),
    "reals": np.concatenate([df_time_to_cancer_dx['reals'], df_time_to_cancer_dx['reals'], df_time_to_cancer_dx['reals']]),
    "times": np.concatenate([df_time_to_cancer_dx['ttcancer'], df_time_to_cancer_dx['ttcancer'], df_time_to_cancer_dx['ttcancer']])
})

# # Placeholder for add_cutoff_strata function
data_to_adjust = add_cutoff_strata(data_to_adjust, by=0.1)

data_to_adjust = pivot_longer_strata(data_to_adjust)

data_to_adjust["reals"] = data_to_adjust["reals"].replace({
    0: "real_negatives",
    2: "real_competing",
    1: "real_positives"
})


data_to_adjust["reals"] = pd.Categorical(data_to_adjust["reals"], categories=["real_negatives", "real_competing", "real_positives"], ordered=True)

# Splitting data by reference group
list_data_to_adjust = {k: v for k, v in data_to_adjust.groupby("reference_group")}


# # Define assumption sets
assumption_sets = [
    {"competing": "excluded", "censored": "excluded"},
    {"competing": "adjusted_as_negative", "censored": "adjusted"},
    {"competing": "adjusted_as_censored", "censored": "adjusted"},
    {"competing": "excluded", "censored": "adjusted"},
    {"competing": "adjusted_as_negative", "censored": "excluded"}
]



adjusted_data_list = []
for reference_group, group_data in list_data_to_adjust.items():
    for assumptions in assumption_sets:
        adjusted_data = extract_aj_estimate_by_assumptions(
            group_data,
            fixed_time_horizons=fixed_time_horizons,
            censoring_assumption=assumptions["censored"],
            competing_assumption=assumptions["competing"]
        )
        adjusted_data["reference_group"] = reference_group
        adjusted_data_list.append(adjusted_data)


final_adjusted_data = pd.concat(adjusted_data_list, ignore_index=True)

aj_data_combinations['strata'] = aj_data_combinations['strata'].astype(str)

final_adjusted_data['strata'] = final_adjusted_data['strata'].astype(str)

aj_data_combinations['reals'] = aj_data_combinations['reals'].astype(str)

final_adjusted_data['reals'] = final_adjusted_data['reals'].astype(str)

categories = ["real_negatives", "real_positives", "real_competing", "real_censored"]
aj_data_combinations['reals'] = pd.Categorical(aj_data_combinations['reals'], categories=categories, ordered=True)
final_adjusted_data['reals'] = pd.Categorical(final_adjusted_data['reals'], categories=categories, ordered=True)

combined_adjusted_data = aj_data_combinations.merge(final_adjusted_data, on=["reference_group", "fixed_time_horizon", "censoring_assumption", "competing_assumption", "reals", "strata"], how='left')




# combined_adjusted_data.dropna(subset=['reals_estimate'])
# # 

# Perform left join between aj_data_combinations and final_adjusted_data on 'strata' and 'reals_estimate'
# only when stratified_by == 'probability_threshold' for aj_data_combinations

aj_data_combinations_prob_threshold = aj_data_combinations[aj_data_combinations['stratified_by'] == 'probability_threshold']

# Convert 'strata' columns to strings
aj_data_combinations_prob_threshold['strata'] = aj_data_combinations_prob_threshold['strata'].astype(str)
final_adjusted_data['strata'] = final_adjusted_data['strata'].astype(str)

combined_adjusted_data = aj_data_combinations_prob_threshold.merge(
    final_adjusted_data[['strata', 'reals', 'reals_estimate']],
    on=['strata', 'reals'],
    how='left'
)


combined_adjusted_data.to_pickle("combined_adjusted_data.pkl")


# aj_data_combinations_prob_threshold = aj_data_combinations[aj_data_combinations['stratified_by'] == 'probability_threshold']

# # Convert 'strata' columns to strings
# aj_data_combinations_prob_threshold['strata'] = aj_data_combinations_prob_threshold['strata'].astype(str)
# final_adjusted_data['strata'] = final_adjusted_data['strata'].astype(str)

# combined_adjusted_data = aj_data_combinations_prob_threshold.merge(
#     final_adjusted_data[['strata', 'reals', 'reals_estimate']],
#     on=['strata', 'reals'],
#     how='left'
# )



# from rtichoke.summary_report.summary_report import render_summary_report


# help(render_summary_report)

# probs = [0.1, 0.4, 0.8]
# reals = [0, 1, 1]
# times = [1, 3, 5]
# render_summary_report(probs, reals, times)


# import subprocess

# def render_quarto_report(probs_vector):
#     # Convert probabilities to a comma-separated string
#     probs_str = ",".join(map(str, probs_vector))
    
#     # Render with Quarto CLI
#     subprocess.run([
#         "quarto", "render", 
#         "summary_report_template.qmd",
#         "-P", f"probs:{probs_str}"
#     ])

# render_quarto_report(probs=[0.1, 0.4, 0.8])

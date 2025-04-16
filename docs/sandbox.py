from lifelines import CoxPHFitter
from lifelines import WeibullAFTFitter
import pandas as pd
import numpy as np
import pickle

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

probs_dict = {
    "thin": pred_thin,
    "full": pred_1_5,
    "aft": preds_aft
}

# The import statement has been moved to the top of the file.

with open('probs_dict.pkl', 'wb') as file:
    pickle.dump(probs_dict, file)
        
with open('reals_dict.pkl', 'wb') as file:
    pickle.dump(df_time_to_cancer_dx['reals']
, file)

with open('times_dict.pkl', 'wb') as file:
    pickle.dump(df_time_to_cancer_dx['ttcancer']
, file)
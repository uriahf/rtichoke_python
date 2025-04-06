import pandas as pd
import matplotlib.pyplot as plt
from lifelines import AalenJohansenFitter

df_time_to_cancer_dx = pd.read_csv(
    "https://raw.githubusercontent.com/ddsjoberg/dca-tutorial/main/data/df_time_to_cancer_dx.csv"
)

reals_mapping = {
    "censor": 0,
    "diagnosed with cancer": 1,
    "dead other causes": 2
}
df_time_to_cancer_dx['reals'] = df_time_to_cancer_dx['cancer_cr'].map(reals_mapping)

ajf_primary = AalenJohansenFitter()
ajf_primary.fit(
    durations=df_time_to_cancer_dx['ttcancer'],
    event_observed=df_time_to_cancer_dx['reals'],
    event_of_interest=1
)

ajf_competing = AalenJohansenFitter()
ajf_competing.fit(
    durations=df_time_to_cancer_dx['ttcancer'],
    event_observed=df_time_to_cancer_dx['reals'],
    event_of_interest=2
)

ajf_competing.predict(1)

type(ajf_competing.predict([1]))

ajf_competing.plot_cumulative_density()
plt.show()
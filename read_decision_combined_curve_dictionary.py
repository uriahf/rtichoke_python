import pandas as pd
from rtichoke.rtichoke_curves.exported_functions import create_plotly_curve
from rtichoke.rtichoke_curves.exported_functions import plot_decision_combined_curve
import json

f = open('C:/Users/CRI_user/Documents/rtichoke_curve_decision_combined_json_array.json')
rtichoke_decision_combined_curve_dict = json.load(f)

print(rtichoke_decision_combined_curve_dict.keys())
# print(rtichoke_decision_combined_curve_dict["group_colors_vec"])

rtichoke_decision_combined_curve_dict["reference_data"]["conventional"] = pd.DataFrame.from_dict(rtichoke_decision_combined_curve_dict["reference_data"]["conventional"]) 
rtichoke_decision_combined_curve_dict["performance_data_ready_for_curve"]["conventional"] = pd.DataFrame.from_dict(rtichoke_decision_combined_curve_dict["performance_data_ready_for_curve"]["conventional"]) 
rtichoke_decision_combined_curve_dict["performance_data_ready_for_curve"]["interventions avoided"] = pd.DataFrame.from_dict(rtichoke_decision_combined_curve_dict["performance_data_ready_for_curve"]["interventions avoided"]) 

print("rtichoke decision combined curve dict keys")
print(rtichoke_decision_combined_curve_dict.keys())

fig_new = plot_decision_combined_curve(
    rtichoke_decision_combined_curve_dict
)

print(len(fig_new.to_dict()["data"]))

# fig_new.to_dict()["data"]

# fig_new

fig_new.show()

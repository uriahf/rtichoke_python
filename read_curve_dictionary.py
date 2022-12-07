import pandas as pd
from rtichoke.rtichoke_curves.exported_functions import create_plotly_curve
import json

f = open('C:/Users/CRI_user/Documents/rtichoke_curve_json_array.json')
rtichoke_curve_dict = json.load(f)

rtichoke_curve_dict["reference_data"] = pd.DataFrame.from_dict(rtichoke_curve_dict["reference_data"]) 
rtichoke_curve_dict["performance_data_ready_for_curve"] = pd.DataFrame.from_dict(rtichoke_curve_dict["performance_data_ready_for_curve"]) 


# print(rtichoke_curve_dict["reference_data"])

# not (rtichoke_curve_dict["reference_data"].empty)

# print("new rtichoke dict")
# print(
# rtichoke_curve_dict["perf_dat_type"] not in ["several models", "several populations"]
# )




fig_new = create_plotly_curve(
    rtichoke_curve_dict
)

fig_new.show()

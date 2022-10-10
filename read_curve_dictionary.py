import pandas as pd
from rtichoke.rtichoke_curves.exported_functions import create_plotly_curve
import json

reference_data = pd.read_csv('C:/Users/CRI_user/Documents/reference_data.csv')
performance_data_ready_for_curve = pd.read_csv('C:/Users/CRI_user/Documents/performance_data_ready_for_curve.csv')


f = open('C:/Users/CRI_user/Documents/group_colors_vec.json')
group_colors_vec = json.load(f)

f = open('C:/Users/CRI_user/Documents/axis_ranges.json')
axis_ranges = json.load(f)


f = open('C:/Users/CRI_user/Documents/rtichoke_curve_json_array.json')
rtichoke_curve_dict = json.load(f)

rtichoke_curve_dict_old =      {
        "reference_data" : reference_data,
        "performance_data_ready_for_curve" : performance_data_ready_for_curve,
        "group_colors_vec": group_colors_vec,
        "axis_ranges" : axis_ranges
    }

rtichoke_curve_dict["reference_data"] = pd.DataFrame.from_dict(rtichoke_curve_dict["reference_data"]) 
rtichoke_curve_dict["performance_data_ready_for_curve"] = pd.DataFrame.from_dict(rtichoke_curve_dict["performance_data_ready_for_curve"]) 



print("new rtichoke dict")
print(
rtichoke_curve_dict["perf_dat_type"] not in ["several models", "several populations"]
)




fig_new = create_plotly_curve(
    rtichoke_curve_dict
)

fig_new.show()

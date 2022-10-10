import pandas as pd
from rtichoke.rtichoke_curves.exported_functions import create_plotly_curve

reference_data = pd.read_csv('C:/Users/CRI_user/Documents/reference_data.csv')
performance_data_ready_for_curve = pd.read_csv('C:/Users/CRI_user/Documents/performance_data_ready_for_curve.csv')


import json
f = open('C:/Users/CRI_user/Documents/group_colors_vec.json')
group_colors_vec = json.load(f)

f = open('C:/Users/CRI_user/Documents/axis_ranges.json')
axis_ranges = json.load(f)

fig_new = create_plotly_curve(
    {
        "reference_data" : reference_data,
        "performance_data_ready_for_curve" : performance_data_ready_for_curve,
        "group_colors_vec": group_colors_vec,
        "axis_ranges" : axis_ranges
    }
)

fig_new.show()


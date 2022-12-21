import requests
import pandas as pd
from rtichoke.rtichoke_curves.exported_functions import create_plotly_curve


def create_lift_curve(probs, reals, stratified_by = "probability_threshold"):
    
    r = requests.post(
   "http://127.0.0.1:6706/create_lift_curve_list",  
   json = {
    "probs" : probs,
    "reals" : reals,
    "stratified_by": stratified_by
        }
    )

    rtichoke_curve_list = r.json()

    rtichoke_curve_list["reference_data"] = pd.DataFrame.from_dict(rtichoke_curve_list["reference_data"]) 
    rtichoke_curve_list["performance_data_ready_for_curve"] = pd.DataFrame.from_dict(rtichoke_curve_list["performance_data_ready_for_curve"]) 

    fig = create_plotly_curve(
    rtichoke_curve_list
    )

    return fig
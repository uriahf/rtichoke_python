import requests
import pandas as pd
from rtichoke.rtichoke_curves.exported_functions import create_plotly_curve

def send_requests_to_rtichoke_r(
        dictionary_to_send,
        url_api,
        endpoint
    ):

    r = requests.post(
        f"{url_api}{endpoint}",
        json = dictionary_to_send
        )
    return r

def create_rtichoke_curve(
    probs, 
    reals, 
    stratified_by = "probability_threshold",
    url_api = "http://localhost:4242/",
    curve = "roc"):

    r = send_requests_to_rtichoke_r(
           dictionary_to_send = {
                "probs": probs,
                "reals": reals,
                "stratified_by": stratified_by
           },
           url_api = url_api,
           endpoint = f"create_{curve}_curve_list" 
        )

    rtichoke_curve_list = r.json()

    rtichoke_curve_list["reference_data"] = pd.DataFrame.from_dict(rtichoke_curve_list["reference_data"]) 
    rtichoke_curve_list["performance_data_ready_for_curve"] = pd.DataFrame.from_dict(rtichoke_curve_list["performance_data_ready_for_curve"]) 

    fig = create_plotly_curve(
    rtichoke_curve_list
    )

    return fig

def plot_rtichoke_curve(
    performance_data,
    url_api = "http://localhost:4242/",
    curve = "roc"):

    r = send_requests_to_rtichoke_r(
           dictionary_to_send = {
                "performance_data": performance_data
           },
           url_api = url_api,
           endpoint = f"plot_{curve}_curve_list" 
        )

    rtichoke_curve_list = r.json()

    rtichoke_curve_list["reference_data"] = pd.DataFrame.from_dict(rtichoke_curve_list["reference_data"]) 
    rtichoke_curve_list["performance_data_ready_for_curve"] = pd.DataFrame.from_dict(rtichoke_curve_list["performance_data_ready_for_curve"]) 

    fig = create_plotly_curve(
    rtichoke_curve_list
    )

    return fig
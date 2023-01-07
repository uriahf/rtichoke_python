import requests
import pandas as pd
from rtichoke.rtichoke_curves.exported_functions import create_plotly_curve
from rtichoke.rtichoke_curves.send_post_request_to_r_rtichoke import create_rtichoke_curve


def create_lift_curve(probs, reals, stratified_by = "probability_threshold", url_api = "http://localhost:4242/"):
    fig = create_rtichoke_curve(
            probs, 
            reals, 
            stratified_by = stratified_by,
            url_api = url_api,
            curve = "lift")
    return fig
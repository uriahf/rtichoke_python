import requests
import pandas as pd
from rtichoke.rtichoke_curves.send_post_request_to_r_rtichoke import send_requests_to_rtichoke_r

def prepare_performance_data(probs, reals, stratified_by = "probability_threshold", url_api = "http://localhost:4242/"):
    
    r = send_requests_to_rtichoke_r(
           dictionary_to_send = {
                "probs": probs,
                "reals": reals,
                "stratified_by": stratified_by
           },
           url_api = url_api,
           endpoint = "prepare_performance_data" 
        )

    performance_data = pd.DataFrame.from_dict(r.json())
    return performance_data
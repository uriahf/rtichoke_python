import requests
import pandas as pd
from rtichoke.helpers.send_post_request_to_r_rtichoke import send_requests_to_rtichoke_r

def create_summary_report(probs, reals, url_api = "http://localhost:4242/"):
    
    r = send_requests_to_rtichoke_r(
           dictionary_to_send = {
                "probs": probs,
                "reals": reals
           },
           url_api = url_api,
           endpoint = "create_summary_report" 
        )
    print(r.json()[0].keys())
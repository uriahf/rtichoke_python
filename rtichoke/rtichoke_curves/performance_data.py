import requests
import pandas as pd

def prepare_performance_data(probs, reals, stratified_by = "probability_threshold"):
    
    r = requests.post(
   "http://127.0.0.1:6706/prepare_performance_data",  
   json = {
    "probs" : probs,
    "reals" : reals,
    "stratified_by": stratified_by 
        }
    )

    performance_data = pd.DataFrame.from_dict(r.json())
    return performance_data
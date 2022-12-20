import pandas as pd
from rtichoke.rtichoke_curves.exported_functions import create_plotly_curve
from rtichoke.rtichoke_curves.roc import create_roc_curve
import json
import requests
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np



lr = LogisticRegression()
x = np.arange(10).reshape(-1, 1)
y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

model = LogisticRegression(solver='liblinear', random_state=0)

model.fit(x, y)

roc_curve = create_roc_curve(
    probs = {'Logistic Regression' :  model.predict_proba(x)[:,1].tolist()},
    reals = {'Logistic Regression' : y.tolist()}
)

roc_curve.show()


r = requests.post(
   "http://127.0.0.1:6706/roc_curve_list",  
   json = {
    "probs" : {'Logistic Regression' :  model.predict_proba(x)[:,1].tolist()},
    "reals" : {'Logistic Regression' : y.tolist()} 
    }
)

rtichoke_curve_list2 = r.json()

rtichoke_curve_list2

print(rtichoke_curve_list2.keys())
rtichoke_curve_list2["reference_data"] = pd.DataFrame.from_dict(rtichoke_curve_list2["reference_data"]) 

print(rtichoke_curve_list2['reference_data'])

print(rtichoke_curve_list2['axes_labels'])

print(rtichoke_curve_list2['group_colors_vec'])

rtichoke_curve_list2["performance_data_ready_for_curve"] = pd.DataFrame.from_dict(rtichoke_curve_list2["performance_data_ready_for_curve"]) 


print(rtichoke_curve_list2["axes_ranges"])



fig_new = create_plotly_curve(
    rtichoke_curve_list2
)

fig_new.show()

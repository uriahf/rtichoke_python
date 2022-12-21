import pandas as pd
from rtichoke.rtichoke_curves.exported_functions import create_plotly_curve
from rtichoke.rtichoke_curves.roc import create_roc_curve
from rtichoke.rtichoke_curves.lift import create_lift_curve
from rtichoke.rtichoke_curves.gains import create_gains_curve
from rtichoke.rtichoke_curves.precision_recall import create_precision_recall_curve
from rtichoke.rtichoke_curves.performance_data import prepare_performance_data


import json
import requests
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np
print(pd.__version__)




lr = LogisticRegression()
x = np.arange(10).reshape(-1, 1)
y = np.array([0, 1, 0, 0, 1, 1, 1, 0, 0, 1])

x_test = np.arange(7).reshape(-1, 1)
y_test = np.array([1, 0, 1, 0, 1, 0, 0])

model = LogisticRegression(solver='liblinear', random_state=0)
lasso = LogisticRegression(solver='liblinear', penalty="l1", random_state=0)



model.fit(x, y)
lasso.fit(x_test, y_test)

roc_curve = create_roc_curve(
    probs = {'Logistic Regression' :  model.predict_proba(x)[:,1].tolist()},
    reals = {'Logistic Regression' : y.tolist()}
)

roc_curve.show()

roc_curve_multiple_models = create_roc_curve(
    probs = {'Logistic Regression' :  model.predict_proba(x)[:,1].tolist(),
             'Lasso' :  lasso.predict_proba(x)[:,1].tolist()},
    reals = {'Outcome' : y.tolist()}
)

roc_curve_multiple_models.show()

roc_curve_multiple_populations = create_roc_curve(
    probs = {'Train' :  model.predict_proba(x)[:,1].tolist(),
             'Test' :  model.predict_proba(x_test)[:,1].tolist()},
    reals = {'Train' : y.tolist(),
             'Test' : y_test.tolist()}
)

roc_curve_multiple_populations.show()

roc_curve_ppcr = create_roc_curve(
    probs = {'Logistic Regression' :  model.predict_proba(x)[:,1].tolist()},
    reals = {'Logistic Regression' : y.tolist()},
    stratified_by="ppcr"
)

roc_curve_ppcr.show()

lift_curve = create_lift_curve(
    probs = {'Logistic Regression' :  model.predict_proba(x)[:,1].tolist()},
    reals = {'Logistic Regression' : y.tolist()}
)

lift_curve.show()

lift_curve_ppcr = create_lift_curve(
    probs = {'Logistic Regression' :  model.predict_proba(x)[:,1].tolist()},
    reals = {'Logistic Regression' : y.tolist()},
    stratified_by="ppcr"
)

lift_curve_ppcr.show()


precision_recall_curve = create_precision_recall_curve(
    probs = {'Logistic Regression' : model.predict_proba(x)[:,1].tolist()},
    reals = {'Logistic Regression' : y.tolist()}
)

precision_recall_curve.show()

precision_recall_curve_ppcr = create_precision_recall_curve(
    probs = {'Logistic Regression' :  model.predict_proba(x)[:,1].tolist()},
    reals = {'Logistic Regression' : y.tolist()},
    stratified_by="ppcr"
)

precision_recall_curve_ppcr.show()

gains_curve = create_gains_curve(
    probs = {'Logistic Regression' :  model.predict_proba(x)[:,1].tolist()},
    reals = {'Logistic Regression' : y.tolist()}
)

gains_curve.show()

gains_curve_ppcr = create_gains_curve(
    probs = {'Logistic Regression' :  model.predict_proba(x)[:,1].tolist()},
    reals = {'Logistic Regression' : y.tolist()},
    stratified_by="ppcr"
)

gains_curve_ppcr.show()


performance_data = prepare_performance_data(
    probs = {'Logistic Regression' :  model.predict_proba(x)[:,1].tolist()},
    reals = {'Logistic Regression' : y.tolist()}
)

performance_data


performance_data_ppcr = prepare_performance_data(
    probs = {'Logistic Regression' :  model.predict_proba(x)[:,1].tolist()},
    reals = {'Logistic Regression' : y.tolist()},
    stratified_by="ppcr"
)

performance_data_ppcr

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

from IPython.display import display
from rtichoke import Rtichoke
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

# create fake data
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import fetch_california_housing
import pandas as pd

data = fetch_california_housing()
X = pd.DataFrame(data["data"], columns=data["feature_names"])
y = (data["target"] > 3).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y)

lr = LogisticRegression().fit(X_train, y_train)

probs = {
    "population1": lr.predict_proba(
        X_test * np.random.normal(loc=1, size=X_test.shape)
    )[:, 1],
    "population2": lr.predict_proba(X_train)[:, 1],
}
reals = {"population1": y_test, "population2": y_train}


r = Rtichoke(probs, reals, by=0.1)


r.plot("ROC", "probability_threshold", filename="temp.html")


# ##

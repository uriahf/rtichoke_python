# Calibration Curves for Multiple Models

The following example is inspired by the [scikit-learn documentation displaying a calibration curve](https://scikit-learn.org/stable/auto_examples/calibration/plot_calibration_curve.html).


# Load data and fit models


``` python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.calibration import CalibratedClassifierCV
import numpy as np
from sklearn.svm import LinearSVC

X, y = make_classification(
    n_samples=10_000, n_features=20, n_informative=2, n_redundant=10, random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.99, random_state=42
)

lr = LogisticRegression(C=1.0)
gnb = GaussianNB()
gnb_isotonic = CalibratedClassifierCV(gnb, cv=2, method="isotonic")
gnb_sigmoid = CalibratedClassifierCV(gnb, cv=2, method="sigmoid")

lr.fit(X_train, y_train)
gnb.fit(X_train, y_train)
gnb_isotonic.fit(X_train, y_train)
gnb_sigmoid.fit(X_train, y_train)

y_proba_lr = lr.predict_proba(X_test)[:, 1]
y_proba_gnb = gnb.predict_proba(X_test)[:, 1]
y_proba_gnb_isotonic = gnb_isotonic.predict_proba(X_test)[:, 1]
y_proba_gnb_sigmoid = gnb_sigmoid.predict_proba(X_test)[:, 1]

class NaivelyCalibratedLinearSVC(LinearSVC):
    def fit(self, X, y):
        super().fit(X, y)
        df = self.decision_function(X)
        self.df_min_ = df.min()
        self.df_max_ = df.max()

    def predict_proba(self, X):
        df = self.decision_function(X)
        calibrated_df = (df - self.df_min_) / (self.df_max_ - self.df_min_)
        proba_pos_class = np.clip(calibrated_df, 0, 1)
        proba_neg_class = 1 - proba_pos_class
        return np.c_[proba_neg_class, proba_pos_class]

svc = NaivelyCalibratedLinearSVC(max_iter=10_000)
svc_isotonic = CalibratedClassifierCV(svc, cv=2, method="isotonic")
svc_sigmoid = CalibratedClassifierCV(svc, cv=2, method="sigmoid")

svc.fit(X_train, y_train)
svc_isotonic.fit(X_train, y_train)
svc_sigmoid.fit(X_train, y_train)

y_proba_svc = svc.predict_proba(X_test)[:, 1]
y_proba_svc_isotonic = svc_isotonic.predict_proba(X_test)[:, 1]
y_proba_svc_sigmoid = svc_sigmoid.predict_proba(X_test)[:, 1]
```


# Gaussian Naive Bayes


``` python
from rtichoke import create_calibration_curve

create_calibration_curve(
    probs={
        "Logistic": y_proba_lr,
        "Naive Bayes": y_proba_gnb,
        "Naive Bayes + Isotonic": y_proba_gnb_isotonic,
        "Naive Bayes + Sigmoid": y_proba_gnb_sigmoid,
    },
    reals=y_test
).show(config={"displayModeBar": False, "displaylogo": False})
```


# Linear SVC


``` python
create_calibration_curve(
    probs={
        "Logistic": y_proba_lr,
        "SVC": y_proba_svc,
        "SVC + Isotonic": y_proba_svc_isotonic,
        "SVC + Sigmoid": y_proba_svc_sigmoid,
    },
    reals=y_test
).show(config={"displayModeBar": False, "displaylogo": False})
```


This reproduces the core comparison from the scikit-learn example while keeping the rtichoke rendering as the Python example itself.

"""Functions to create performance data tables"""

from typing import Union
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from tqdm import tqdm


def prepare_performance_data(self: object, stratified_by: float):
    """
    User's function to produce performance data table for probs/reals.
    probs/reals may represent one probs vs. one reals, several probs vs. one real,
    or several probs vs. several reals.

    Args:
        probs (list, np.array, pd.Series, or dict): an array of probabilities or a dictionary
                                                    {'pop_name': array of probabilities}
        reals (list, np.array, pd.Series, or dict): an array of binary results or a dictionary
                                                    {'pop_name': arary of binary results}
        by (float, optional): argument to set the distance between explored threshold probabilities.
                                Defaults to 0.01.
        stratified_by (string, optional): must be either "probability_threshold" or "ppcr".
                                Defaults to "probability_threshold".

    Returns:
        pd.DataFrame: a dataframe with performance metrics
    """
    if isinstance(self.probs, dict) and isinstance(self.reals, dict):
        assert (
            self.probs.keys() == self.reals.keys()
        ), "When sending dictionaries, probs and reals must have the same keys"

        return pd.concat(
            [
                self.prepare_performance_table(
                    probs=self.probs[key],
                    reals=self.reals[key],
                    by=self.by,
                    stratified_by=stratified_by,
                    pop_name=key,
                )
                for key in self.probs.keys()
            ]
        )

    if isinstance(self.probs, dict) and isinstance(
        self.reals, (list, np.ndarray, pd.Series)
    ):
        return pd.concat(
            [
                self.prepare_performance_table(
                    probs=self.probs[key],
                    reals=self.reals,
                    by=self.by,
                    stratified_by=stratified_by,
                    pop_name=key,
                )
                for key in self.probs.keys()
            ]
        )

    if isinstance(self.probs, (list, np.ndarray, pd.Series)) and isinstance(
        self.reals, (list, np.ndarray, pd.Series)
    ):
        return self.prepare_performance_table(
            probs=self.probs, reals=self.reals, by=self.by, stratified_by=stratified_by
        )

    raise ValueError("Wrong inputs provided for probs and reals")


def prepare_performance_table(
    self: object,
    probs: Union[list, np.ndarray, dict],
    reals: Union[list, np.ndarray, dict],
    by: float,
    stratified_by: str,
    pop_name: str = "pop1",
):
    """Generate performance table for a single set of probs and reals.

    Args:
        probs (list, np.array, pd.Series): an array of probabilities
        reals (list, np.array, pd.Series): an array of true values (0's or 1's)
        by (float, optional): argument to set the distance between explored
                                        threshold probabilities. Defaults to 0.01.
        stratified_by (string, optional): must be either "probability_threshold" or "ppcr".
                                        Defaults to "probability_threshold".
        pop_name (str, optional): A population name, when asking for performance
                                    metrics for several populations. Defaults to 'pop1'.

    Returns:
        pd.DataFrame: a dataframe with performance metrics
    """

    # update prevalence and N:
    self.prevalence.update({pop_name: reals.mean()})
    self.N.update({pop_name: len(reals)})

    # convert inputs to np.arrays
    probs = np.array(probs)
    reals = np.array(reals)

    # verify inputs
    self.validate_inputs(probs, reals)
    # decimals = len(str(self.by).split(".")[1].rstrip("0"))

    # define probabilty thresholds
    prob_thresholds = np.append(np.arange(0, 1, by), 1)

    # if ppcr is required, adjust probability threholds accordingly.
    if stratified_by == "ppcr":
        ppcr = np.append(np.arange(0, 1, by), 1)
        prob_thresholds = np.array([np.quantile(probs, p) for p in prob_thresholds])
        prob_thresholds[0] = 0.0
    else:
        ppcr = []

    # define performance table
    performance_table = {
        "Population": [],
        "probability_threshold": prob_thresholds,
        "ppcr": ppcr,
        "predicted_positives": [],
        "TP": [],
        "FP": [],
        "FN": [],
        "TN": [],
    }

    # run over all probability thresholds and calculate confusion matrix
    for p in tqdm(
        prob_thresholds, desc="Calculating performance data", leave=False, delay=0.5
    ):
        preds = (probs > p).astype(int)
        if stratified_by == "probability_threshold":
            performance_table["ppcr"].append(preds.mean())
        performance_table["predicted_positives"].append(preds.sum())

        tn, fp, fn, tp = confusion_matrix(reals, preds).ravel()
        performance_table["TP"].append(tp)
        performance_table["FP"].append(fp)
        performance_table["FN"].append(fn)
        performance_table["TN"].append(tn)

    # define additional metrics
    performance_table["Population"] = [pop_name] * len(prob_thresholds)
    performance_table = pd.DataFrame(performance_table)
    performance_table["Sensitivity"] = performance_table["TP"] / (
        performance_table["TP"] + performance_table["FN"]
    )
    performance_table["Specificity"] = performance_table["TN"] / (
        performance_table["TN"] + performance_table["FP"]
    )
    performance_table["FPR"] = 1 - performance_table["Specificity"]

    performance_table["PPV"] = (
        performance_table["TP"] / (performance_table["TP"] + performance_table["FP"])
    ).fillna(1.0)
    performance_table["NPV"] = performance_table["TN"] / (
        performance_table["TN"] + performance_table["FN"]
    )
    performance_table["lift"] = performance_table["PPV"] / reals.mean()
    performance_table["Net_benefit"] = performance_table[
        "Sensitivity"
    ] * reals.mean() - (1 - performance_table["Specificity"]) * (1 - reals.mean()) * (
        performance_table["probability_threshold"]
        / (1 - performance_table["probability_threshold"])
    )

    return (
        performance_table
        if stratified_by == "probability_threshold"
        else performance_table.iloc[::-1]
    )

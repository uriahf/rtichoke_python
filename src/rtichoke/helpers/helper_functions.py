from datetime import datetime
import numpy as np


def tprint(string):
    now = datetime.strftime(datetime.now(), "%Y-%m-%d %H:%M:%S")
    print(now + " - " + string)


def select_data_table(self, x, y, stratification="probability_threshold"):
    df = (
        self.performance_table_pt
        if stratification == "probability_threshold"
        else self.performance_table_ppcr
    )
    cols = list(
        set(
            ["Population", "predicted_positives", "probability_threshold", "ppcr", x, y]
        )
    )
    return df[cols]


def modified_calibration_curve(
    self,
    reals,
    probs,
    n_bins=10,
    strategy="quantile",
):
    """A modified version of sklearn.calibration.calibration_curve (https://scikit-learn.org/stable/modules/generated/sklearn.calibration.calibration_curve.html),
    to return number over cases in each bin.


    Args:
        reals (array-like of shape (n_samples,)): True targets
        probs (array-like of shape (n_samples,)): Probabilities of the positive class.
        n_bins (int, optional): Number of bins to discretize the [0, 1] interval.
                                A bigger number requires more data.
                                Defaults to 10.
        strategy (str, optional): Strategy used to define the widths of the bins.
        uniform: The bins have identical widths.
        quantile: The bins have the same number of samples and depend on probs (Default).

    Returns:
        prob_true: The proportion of samples whose class is the positive class, in each bin (fraction of positives).
        prob_pred: The mean predicted probability in each bin.
        bin_sums: number of cases predicted as positive in each bin.
        bin_true: number of actual positive in each bin.
        bin_total: number of cases in each bin.
    """
    if strategy == "quantile":  # Determine bin edges by distribution of data
        quantiles = np.linspace(0, 1, n_bins + 1)
        bins = np.percentile(probs, quantiles * 100)
    elif strategy == "uniform":
        bins = np.linspace(0.0, 1.0, n_bins + 1)
    else:
        raise ValueError(
            "Invalid entry to 'strategy' input. Strategy "
            "must be either 'quantile' or 'uniform'."
        )

    binids = np.searchsorted(bins[1:-1], probs)

    bin_sums = np.bincount(binids, weights=probs, minlength=len(bins))
    bin_true = np.bincount(binids, weights=reals, minlength=len(bins))
    bin_total = np.bincount(binids, minlength=len(bins))

    nonzero = bin_total != 0
    prob_true = bin_true[nonzero] / bin_total[nonzero]
    prob_pred = bin_sums[nonzero] / bin_total[nonzero]

    return (
        prob_true,
        prob_pred,
        bin_sums[nonzero].astype(int),
        bin_true[nonzero].astype(int),
        bin_total[nonzero].astype(int),
    )

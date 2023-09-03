"""Functions to create data for calibration plots"""

import numpy as np
import pandas as pd


def prepare_calibration_data(self: object, n_bins: int, strategy: str):
    """
    User's function to produce performance data table for probs/reals.
    probs/reals may represent one probs vs. one reals, several probs vs. one real,
    or several probs vs. several reals.

    Args:
        probs (list, np.array, pd.Series, or dict):
            an array of probabilities or a dictionary {'pop_name': array of probabilities}
        reals (list, np.array, pd.Series, or dict):
            an array of binary results or a dictionary {'pop_name': arary of binary results}

    Returns:
        pd.DataFrame: a dataframe with performance metrics
    """
    if isinstance(self.probs, dict) and isinstance(self.reals, dict):
        assert (
            self.probs.keys() == self.reals.keys()
        ), "When sending dictionaries, probs and reals must have the same keys"

        return pd.concat(
            [
                self.prepare_calibration_table(
                    probs=self.probs[key],
                    reals=self.reals[key],
                    n_bins=n_bins,
                    strategy=strategy,
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
                self.prepare_calibration_table(
                    probs=self.probs[key],
                    reals=self.reals,
                    n_bins=n_bins,
                    strategy=strategy,
                    pop_name=key,
                )
                for key in self.probs.keys()
            ]
        )

    if isinstance(self.probs, (list, np.ndarray, pd.Series)) and isinstance(
        self.reals, (list, np.ndarray, pd.Series)
    ):
        return self.prepare_calibration_table(
            probs=self.probs,
            reals=self.reals,
            n_bins=n_bins,
            strategy=strategy,
        )

    raise ValueError("Wrong inputs provided for probs and reals")


def prepare_calibration_table(
    self: object,
    probs: [list | np.array],
    reals: [list | np.array],
    n_bins: int,
    strategy: str,
    pop_name="pop1",
) -> pd.DataFrame:
    """Generate calibration data table for a single set of probs and reals.

    Args:
        probs (list, np.array, pd.Series): an array of probabilities
        reals (list, np.array, pd.Series): an array of true values (0's or 1's)
        pop_name (str, optional): A population name, when asking for performance
                                  metrics for several populations.
                                  Defaults to 'pop1'.

    Returns:
        pd.DataFrame: a dataframe with calibration data
    """

    # convert inputs to np.arrays
    probs = np.array(probs)
    reals = np.array(reals)

    # verify inputs
    self.validate_inputs(probs, reals)
    (
        prob_true,
        prob_pred,
        pred_pos,
        actual_pos,
        total_cases,
    ) = self.modified_calibration_curve(
        reals=reals, probs=probs, n_bins=n_bins, strategy=strategy
    )

    return pd.DataFrame(
        {
            "Population": [pop_name] * len(prob_true),
            "prob_true": prob_true,
            "prob_pred": prob_pred,
            "pred_pos": pred_pos,
            "actual_pos": actual_pos,
            "total_cases": total_cases,
            "total_N": [self.N[pop_name]] * len(prob_true),
        }
    )

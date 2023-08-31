"""script for validating inputs to rtichoke methods"""


def validate_inputs(_, probs, reals):
    """A mother-function to run all other validation functions

    Args:
        probs (np.array): an array of probabilities
        reals (np.array): an array of true values (0's or 1's)
        by (float): argument to set the distance between explored threshold probabilities
        stratified_by (string): must be either "probability_threshold" or "ppcr"
    """
    check_probs(probs)
    check_probs_vs_reals(probs, reals)
    check_reals(reals)


def check_probs(probs):
    """
    Validate probs by ensuring all values are between [0, 1]

    Args:
        probs (np.array): an array of probabilities

    Raises:
        ValueError: when validation fails

    Returns:
        Boolean: True when validation passed (else raise exception)
    """
    if min(probs) < 0 or max(probs) > 1:
        raise ValueError("Probs must be within [0, 1]")


def check_probs_vs_reals(probs, reals):
    """
    Validate probs vs. reals:
    1. probs and reals must have the same shape
    2. at least two values should be included in each array

    Args:
        probs (np.array): an array of probabilities
        reals (np.array): an array of true values (0's or 1's)

    Raises:
        ValueError: when either validation fails

    Returns:
        Boolean: True when validation passed (else raise exception)
    """
    if probs.shape != reals.shape:
        raise ValueError(
            f"Probs and reals shapes are inconsistent ({probs.shape} and {reals.shape})"
        )
    elif len(probs) < 2:
        raise ValueError("At least two entries should be included reals and probs")


def check_reals(reals):
    """
    Validate reals consist of only 0's and 1's, including positive and negative examples

    Args:
        reals (np.array): an array of true values (0's or 1's)

    Raises:
        ValueError: when validation fails

    Returns:
        Boolean: True when validation passed (else raise exception)
    """
    if set(reals) != {0, 1}:
        raise ValueError("Reals must include only 0's and 1's")


def check_by(self):
    """
    Validate `by` argument is between 0 and 0.5

    Args:
        by (float): argument to set the distance between explored threshold probabilities

    Raises:
        ValueError: when validation fails

    Returns:
        Boolean: True when validation passed (else raise exception)
    """
    if not (isinstance(self.by, float) and (self.by > 0) and (self.by <= 0.5)):
        raise ValueError("Argument `by` must be a float,  0 > `by` <= 0.5")


def validate_plot_inputs(_, curve_type, stratification):
    """This function runs child validation functions to ensure proper plotting inputs

    Args:
        curve_type (str): Available plots: "ROC", "LIFT", "PR", "NB", or "calibration"
        stratification (str, optional): Stratifiction method ("PPCR" or "probability_threshold").
                                        Defaults to "probability_threshold".
    """
    check_plot_curve_type(curve_type)
    check_plot_stratification(stratification)


def check_plot_curve_type(curve_type):
    """A method to verify requested curve_type is available.

    Args:
        curve_type (str): Available plots: "ROC", "LIFT", "PR", "NB", or "calibration"

    Raises:
        ValueError: when `curve_type` is not one of the available plots.
    """
    available_plots = ["ROC", "LIFT", "PR", "NB", "calibration"]
    if curve_type not in available_plots:
        raise ValueError(
            f"curve_type {curve_type} not recognized. Supported curves :{available_plots}"
        )


def check_plot_stratification(stratification):
    """A method to verify stratification method

    Args:
        stratification (str, optional): Stratifiction method ("PPCR" or "probability_threshold")

    Raises:
        ValueError: when `stratification` is not "PPCR" or "probability_threshold".
    """
    if stratification not in ["probability_threshold", "ppcr"]:
        raise ValueError(
            "stratification has to be wither probability_threshold or ppcr"
        )

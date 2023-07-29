def validate_inputs(self, probs, reals):
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
        Exception: when validation fails

    Returns:
        Boolean: True when validation passed (else raise exception)
    """
    if min(probs) < 0 or max(probs) > 1:
        raise Exception("Probs must be within [0, 1]")
    pass


def check_probs_vs_reals(probs, reals):
    """
    Validate probs vs. reals:
    1. probs and reals must have the same shape
    2. at least two values should be included in each array

    Args:
        probs (np.array): an array of probabilities
        reals (np.array): an array of true values (0's or 1's)

    Raises:
        Exception: when either validation fails

    Returns:
        Boolean: True when validation passed (else raise exception)
    """
    if probs.shape != reals.shape:
        raise Exception(
            f"Probs and reals shapes are inconsistent ({probs.shape} and {reals.shape})"
        )
    elif len(probs) < 2:
        raise Exception("At least two entries should be included reals and probs")
    pass


def check_reals(reals):
    """
    Validate reals consist of only 0's and 1's, including positive and negative examples

    Args:
        reals (np.array): an array of true values (0's or 1's)

    Raises:
        Exception: when validation fails

    Returns:
        Boolean: True when validation passed (else raise exception)
    """
    if set(reals) != {0, 1}:
        raise Exception(f"Reals must include only 0's and 1's")
    pass


def check_by(self):
    """
    Validate `by` argument is between 0 and 0.5

    Args:
        by (float): argument to set the distance between explored threshold probabilities

    Raises:
        Exception: when validation fails

    Returns:
        Boolean: True when validation passed (else raise exception)
    """
    if not (isinstance(self.by, float) and (self.by > 0) and (self.by <= 0.5)):
        raise Exception(f"Argument `by` must be a float,  0 > `by` <= 0.5")
    pass


def validate_plot_inputs(self, curve_type, stratification):
    check_plot_curve_type(curve_type)
    check_plot_stratification(stratification)


def check_plot_curve_type(curve_type):
    available_plots = ["ROC", "LIFT", "PR", "NB"]
    if curve_type not in available_plots:
        raise Exception(
            f"curve_type {curve_type} not recognized. Supported curves :{available_plots}"
        )
    pass


def check_plot_stratification(stratification):
    if stratification not in ["probability_threshold", "ppcr"]:
        raise Exception(
            f"stratification has to be wither probability_threshold or ppcr"
        )
    pass
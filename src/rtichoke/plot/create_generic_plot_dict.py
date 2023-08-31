""" Generic plotting dict"""

plot_dicts = {
    "ROC": {
        "x": "FPR",
        "y": "Sensitivity",
        "reference": {"x": [0, 1], "y": [0, 1]},
        "xlabel": "1-Specificity",
        "ylabel": "Sensitivity",
        "hover_info": [
            "FPR",
            "Sensitivity",
        ],
    },
    "LIFT": {
        "x": "ppcr",
        "y": "lift",
        "reference": {"x": [0, 1], "y": [1, 1]},
        "xlabel": "ppcr",
        "ylabel": "lift",
        "hover_info": [
            "lift",
        ],
    },
    "gains": {
        "x": "ppcr",
        "y": "Sensitivity",
        "reference": {"x": [0, 1], "y": [1, 1]},
        "xlabel": "ppcr",
        "ylabel": "Sensitivity",
        "hover_info": [
            "Sensitivity",
        ],
    },
    "PR": {
        "x": "PPV",
        "y": "Sensitivity",
        "reference": {"x": [0, 0], "y": [0, 0]},
        "xlabel": "Precision",
        "ylabel": "Recall",
        "hover_info": [
            "PPV",
            "Sensitivity",
        ],
    },
    "NB": {
        "x": "probability_threshold",
        "y": "Net_benefit",
        "reference": {"x": [0, 0], "y": [0, 0]},
        "xlabel": "Prob. threshold",
        "ylabel": "Net Benefit",
        "hover_info": [
            "Net_benefit",
        ],
    },
    "calibration": {
        "x": "prob_pred",
        "y": "prob_true",
        "reference": {"x": [0, 1], "y": [0, 1]},
        "xlabel": "Predicted",
        "ylabel": "Observed",
    },
}


def create_generic_plot_dict(curve_type, stratification):
    """returns a generic plot dict, according to curve_type and stratification method

    Args:
        curve_type (str): One of the available curve types
        stratification (str): stratification method

    Returns:
        dictionary: generic plot dictionary to enable plotting
    """
    generic_plot_dict = plot_dicts[curve_type]
    generic_plot_dict["curve_type"] = curve_type
    generic_plot_dict["stratification"] = stratification
    return generic_plot_dict

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
    "PR": {
        "x": "PPV",
        "y": "Sensitivity",
        "reference": {"x": [0, 0], "y": [0, 0]},
        "xlabel": "Precision",
        "ylabel": "Recall",
        "hover_info": [
            "Precision",
            "Recall",
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
}


def create_generic_plot_dict(curve_type, stratification):
    generic_plot_dict = plot_dicts[curve_type]
    generic_plot_dict["curve_type"] = curve_type
    generic_plot_dict["stratification"] = stratification
    return generic_plot_dict

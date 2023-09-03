"""Plotting module for rtichoke"""

from .bokeh.plot_bokeh import plot_bokeh
from .create_generic_plot_dict import create_generic_plot_dict


def plot(
    self: object,
    curve_type: str,
    stratification: str = "probability_threshold",
    filename: str = None,
    api: str = "bokeh",
):
    """A method to fetch generic plot dict and call the specific plotting API

    Args:
        curve_type (str): Curve type to produce ("ROC", "LIFT", "PR", "NB", or "calibration")
        stratification (str, optional): Stratifiction method ("PPCR" or "probability_threshold").
                                        Defaults to "probability_threshold".
        filename (str, optional): Filename for saving plot. Defaults to None.
        api (str, optional): Which plotting API to use. Currently only "bokeh" is available.
                            Defaults to "bokeh".

    Returns: plot object
    """
    self.validate_plot_inputs(curve_type, stratification)

    generic_plot_dict = create_generic_plot_dict(curve_type, stratification)

    if api == "bokeh":
        return plot_bokeh(self, generic_plot_dict, filename)

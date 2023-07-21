from .bokeh.plot_bokeh import plot_bokeh
from .create_generic_plot_dict import create_generic_plot_dict


def plot(
    self, curve_type, stratification="probability_threshold", filename=None, api="bokeh"
):
    self.validate_plot_inputs(curve_type, stratification)

    generic_plot_dict = create_generic_plot_dict(curve_type, stratification)

    if api == "bokeh":
        return plot_bokeh(self, generic_plot_dict, filename)

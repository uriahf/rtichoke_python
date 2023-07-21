"""rtichoke is a package for interactive vizualization of performance metrics
"""

from importlib.metadata import version

# __version__ = version("rtichoke")

from .helpers.helper_functions import *


class Rtichoke:
    # import methods
    from .performance_data._performance_data import (
        prepare_performance_data,
        prepare_performance_table,
    )
    from .helpers.validations import validate_inputs, validate_plot_inputs, check_by
    from .helpers.helper_functions import tprint, select_data_table
    from .plot.plotting import plot

    def __init__(self, probs=None, reals=None, by=0.01):
        super().__init__()

        self.probs = probs
        self.reals = reals
        self.by = by
        self.performance_table_pt = None
        self.performance_table_ppcr = None
        self.prevalence = {}

        self.check_by()

        tprint("Calculating performance table stratified by probability threshold")
        self.performance_table_pt = self.prepare_performance_data(
            stratified_by="probability_threshold"
        )

        tprint("Calculating performance table stratified by ppcr")
        self.performance_table_ppcr = self.prepare_performance_data(
            stratified_by="ppcr"
        )

"""rtichoke is a package for interactive vizualization of performance metrics
"""

from importlib.metadata import version

# __version__ = version("rtichoke")

from .helpers.helper_functions import *


class Rtichoke:
    # import methods
    from .performance_data.prepare_performance_data import (
        prepare_performance_data,
        prepare_performance_table,
    )
    from .performance_data.prepare_calibration_data import (
        prepare_calibration_data,
        prepare_calibration_table,
    )

    from .helpers.validations import validate_inputs, validate_plot_inputs, check_by
    from .helpers.helper_functions import (
        select_data_table,
        modified_calibration_curve,
    )
    from .plot.plotting import plot

    def __init__(
        self, probs=None, reals=None, by=0.01, cal_n_bins=10, cal_strategy="quantile"
    ):
        super().__init__()

        self.probs = probs
        self.reals = reals
        self.by = by
        self.performance_table_pt = None
        self.performance_table_ppcr = None
        self.calibration_table = None
        self.prevalence = {}
        self.N = {}

        self.check_by()

        tprint("Calculating performance table stratified by probability threshold")
        self.performance_table_pt = self.prepare_performance_data(
            stratified_by="probability_threshold"
        )

        tprint("Calculating performance table stratified by ppcr")
        self.performance_table_ppcr = self.prepare_performance_data(
            stratified_by="ppcr"
        )

        tprint("Calculating calibration data")
        self.calibration_table = self.prepare_calibration_data(
            n_bins=cal_n_bins, strategy=cal_strategy
        )

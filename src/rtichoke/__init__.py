"""rtichoke is a package for interactive vizualization of performance metrics
"""

from importlib.metadata import version

# __version__ = version("rtichoke")

# from rtichoke.discrimination.gains import create_gains_curve
# from rtichoke.discrimination.gains import plot_gains_curve

# from rtichoke.calibration.calibration import create_calibration_curve

# from rtichoke.utility.decision import create_decision_curve
# from rtichoke.utility.decision import plot_decision_curve

# from rtichoke.performance_data.performance_data import prepare_performance_data
# from rtichoke.performance_data.calculate_performance_data import PerformanceData

# from rtichoke.summary_report.summary_report import create_summary_report

# from .rtichoke import Rtichoke
# from .performance_data._performance_data import *
# from .helpers.validations import *

from .helpers.helper_functions import *


class Rtichoke:
    # import methods
    from .performance_data._performance_data import (
        prepare_performance_data,
        prepare_performance_table,
    )
    from .helpers.validations import validate_inputs, validate_plot_inputs, check_by
    from .plot._plot import plot, select_data_table

    def __init__(self, probs=None, reals=None, by=0.01):
        super().__init__()

        self.probs = probs
        self.reals = reals
        self.by = by
        self.performance_table_pt = None
        self.performance_table_ppcr = None
        self.prevalence = {}

        self.check_by()

        print_with_time(
            "Calculating performance table stratified by probability threshold"
        )
        self.performance_table_pt = self.prepare_performance_data(
            stratified_by="probability_threshold"
        )

        print_with_time("Calculating performance table stratified by ppcr")
        self.performance_table_ppcr = self.prepare_performance_data(
            stratified_by="ppcr"
        )

"""rtichoke is a package for interactive vizualization of performance metrics
"""
from typing import Union
import numpy as np
from .helpers.helper_functions import tprint


class Rtichoke:
    """Main Rtichoke class"""

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
        self,
        probs: Union[list, np.ndarray, dict],
        reals: Union[list, np.ndarray, dict],
        by: float = 0.01,
        cal_n_bins: int = 100,
        cal_strategy: str = "quantile",
    ):
        """Rtichoke init method to generate an Rtichoke object

        Args:
            probs (tuple[list  |  np.array]): list or array of probabilities
            reals (tuple[list  |  np.array]): list or array of true outcomes (0's or 1's)
            by (float, optional): a float number indicating the spacing between
                                  probability thresholds. Defaults to 0.01.
            cal_n_bins (int, optional): define how many bins to have in calibration histogram.
                                  Defaults to 100.
            cal_strategy (str, optional): calibration binning method. Defaults to "quantile".

        For details about cal_n_bins and cal_strategy see:
        https://scikit-learn.org/stable/modules/generated/sklearn.calibration.calibration_curve.html
        """

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

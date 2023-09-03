"""rtichoke is a package for interactive vizualization of performance metrics
"""

class Rtichoke:
    """Main Rtichoke class"""

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

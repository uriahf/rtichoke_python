"""rtichoke is a package for interactive vizualization of performance metrics
"""
from importlib.metadata import version

# __version__ = version("rtichoke")

from .helpers.helper_functions import *
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

from .rtichoke import Rtichoke

"""rtichoke is a package for interactive vizualization of performance metrics"""

from importlib.metadata import version

__version__ = version("rtichoke")

from rtichoke.discrimination.roc import create_roc_curve as create_roc_curve
from rtichoke.discrimination.roc import plot_roc_curve as plot_roc_curve

from rtichoke.discrimination.lift import create_lift_curve as create_lift_curve
from rtichoke.discrimination.lift import plot_lift_curve as plot_lift_curve

from rtichoke.discrimination.precision_recall import (
    create_precision_recall_curve as create_precision_recall_curve,
)
from rtichoke.discrimination.precision_recall import (
    plot_precision_recall_curve as plot_precision_recall_curve,
)

from rtichoke.discrimination.gains import create_gains_curve as create_gains_curve
from rtichoke.discrimination.gains import plot_gains_curve as plot_gains_curve

from rtichoke.calibration.calibration import (
    create_calibration_curve as create_calibration_curve,
)

from rtichoke.utility.decision import create_decision_curve as create_decision_curve
from rtichoke.utility.decision import plot_decision_curve as plot_decision_curve

from rtichoke.performance_data.performance_data import (
    prepare_performance_data as prepare_performance_data,
)

from rtichoke.performance_data.performance_data_times import (
    prepare_performance_data_times as prepare_performance_data_times,
)

from rtichoke.summary_report.summary_report import (
    create_summary_report as create_summary_report,
)

__all__ = [
    "create_roc_curve",
    "plot_roc_curve",
    "create_lift_curve",
    "plot_lift_curve",
    "create_precision_recall_curve",
    "plot_precision_recall_curve",
    "create_gains_curve",
    "plot_gains_curve",
    "create_calibration_curve",
    "create_decision_curve",
    "plot_decision_curve",
    "prepare_performance_data",
    "prepare_performance_data_times",
    "create_summary_report",
]

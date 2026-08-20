import numpy as np
import pytest

from rtichoke.discrimination.gains import create_gains_curve
from rtichoke.discrimination.lift import create_lift_curve
from rtichoke.discrimination.precision_recall import create_precision_recall_curve
from rtichoke.discrimination.roc import create_roc_curve
from rtichoke.utility.decision import create_decision_curve


PROBS = {
    "model_a": np.array([0.1, 0.3, 0.7, 0.9]),
    "model_b": np.array([0.2, 0.4, 0.6, 0.8]),
}
REALS = np.array([0, 0, 1, 1])
CUSTOM_COLORS = ["#111111", "#222222"]


@pytest.mark.parametrize(
    "creator",
    [
        create_roc_curve,
        create_precision_recall_curve,
        create_lift_curve,
        create_gains_curve,
        create_decision_curve,
    ],
)
def test_binary_create_curves_honor_custom_colors(creator):
    fig = creator(
        probs=PROBS,
        reals=REALS,
        by=0.25,
        color_values=CUSTOM_COLORS,
    )

    model_traces = [trace for trace in fig.data if trace.showlegend is True]

    assert len(model_traces) == 2
    assert [trace.line.color for trace in model_traces] == CUSTOM_COLORS


def test_binary_single_model_remains_black_like_r():
    fig = create_roc_curve(
        probs={"model": PROBS["model_a"]},
        reals=REALS,
        by=0.25,
        color_values=["#123456"],
    )

    model_trace = next(trace for trace in fig.data if trace.name == "model")

    assert model_trace.line.color == "#000000"


def test_binary_custom_colors_require_one_per_reference_group():
    with pytest.raises(ValueError, match="one color per reference group"):
        create_roc_curve(
            probs=PROBS,
            reals=REALS,
            by=0.25,
            color_values=["#111111"],
        )

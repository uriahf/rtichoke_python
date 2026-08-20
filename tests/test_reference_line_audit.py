import numpy as np
import pytest

from rtichoke import (
    create_decision_curve,
    create_gains_curve,
    create_lift_curve,
    create_precision_recall_curve,
    create_gains_curve_times,
)


def _binary_inputs():
    probs = {
        "population_a": np.array([0.05, 0.15, 0.35, 0.55, 0.75, 0.95]),
        "population_b": np.array([0.10, 0.30, 0.50, 0.70]),
    }
    reals = {
        "population_a": np.array([1, 0, 1, 0, 1, 0]),
        "population_b": np.array([1, 1, 0, 1]),
    }
    return probs, reals


def _visible_trace(fig, name):
    matches = [trace for trace in fig.data if trace.name == name and trace.visible is not False]
    assert len(matches) == 1
    return matches[0]


def _y_at_x(trace, x):
    matches = [float(y) for tx, y in zip(trace.x, trace.y) if float(tx) == pytest.approx(x)]
    assert len(matches) == 1
    return matches[0]


def test_binary_precision_recall_reference_is_population_specific():
    probs, reals = _binary_inputs()
    fig = create_precision_recall_curve(probs, reals, by=0.1)

    a = _visible_trace(fig, "random_guess_population_a")
    b = _visible_trace(fig, "random_guess_population_b")

    assert float(a.y[0]) == pytest.approx(3 / 6)
    assert float(b.y[0]) == pytest.approx(3 / 4)


def test_binary_lift_reference_is_population_specific():
    probs, reals = _binary_inputs()
    fig = create_lift_curve(probs, reals, by=0.1)

    a = _visible_trace(fig, "perfect_model_population_a")
    b = _visible_trace(fig, "perfect_model_population_b")

    assert float(a.y[0]) == pytest.approx(2.0)
    assert float(b.y[0]) == pytest.approx(4 / 3)


def test_binary_decision_reference_is_population_specific():
    probs, reals = _binary_inputs()
    fig = create_decision_curve(
        probs, reals, by=0.1, min_p_threshold=0.1, max_p_threshold=0.9
    )

    a = _visible_trace(fig, "treat_all_population_a")
    b = _visible_trace(fig, "treat_all_population_b")

    x = float(a.x[0])
    assert x == pytest.approx(0.1)
    assert float(a.x[-1]) == pytest.approx(0.9)
    assert float(b.x[0]) == pytest.approx(x)
    assert float(a.y[0]) == pytest.approx(0.5 - 0.5 * x / (1 - x))
    assert float(b.y[0]) == pytest.approx(0.75 - 0.25 * x / (1 - x))


def test_binary_interventions_avoided_honors_threshold_range():
    probs, reals = _binary_inputs()
    fig = create_decision_curve(
        probs,
        reals,
        decision_type="interventions avoided",
        by=0.1,
        min_p_threshold=0.1,
        max_p_threshold=0.9,
    )

    nonempty_visible = [
        trace for trace in fig.data if trace.visible is not False and len(trace.x) > 0
    ]
    assert nonempty_visible
    for trace in nonempty_visible:
        assert min(float(x) for x in trace.x) >= pytest.approx(0.1)
        assert max(float(x) for x in trace.x) <= pytest.approx(0.9)


def test_binary_gains_reference_is_population_specific():
    probs, reals = _binary_inputs()
    fig = create_gains_curve(probs, reals, by=0.1)

    a = _visible_trace(fig, "perfect_model_population_a")
    b = _visible_trace(fig, "perfect_model_population_b")

    assert _y_at_x(a, 0.01) == pytest.approx(0.01 / 0.5)
    assert _y_at_x(b, 0.01) == pytest.approx(0.01 / 0.75)


def test_time_gains_reference_is_population_and_horizon_specific():
    probs = {
        "population_a": np.array([0.05, 0.15, 0.35, 0.55, 0.75, 0.95]),
        "population_b": np.array([0.10, 0.30, 0.50, 0.70]),
    }
    reals = {
        "population_a": np.array([1, 0, 1, 0, 1, 0]),
        "population_b": np.array([1, 1, 0, 1]),
    }
    times = {
        "population_a": np.array([3.0, 11.0, 7.0, 12.0, 13.0, 14.0]),
        "population_b": np.array([4.0, 8.0, 11.0, 12.0]),
    }
    fig = create_gains_curve_times(
        probs, reals, times, fixed_time_horizons=[5.0, 10.0], by=0.1
    )

    traces = {(t.name, t.visible): t for t in fig.data}
    assert _y_at_x(traces[("perfect_model_population_a", True)], 0.01) == pytest.approx(0.06)
    assert _y_at_x(traces[("perfect_model_population_b", True)], 0.01) == pytest.approx(0.04)
    assert _y_at_x(traces[("perfect_model_population_a", False)], 0.01) == pytest.approx(0.03)
    assert _y_at_x(traces[("perfect_model_population_b", False)], 0.01) == pytest.approx(0.02)

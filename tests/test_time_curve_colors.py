import pytest

from rtichoke.processing.time_reference_lines import _apply_color_values_times


def _curve_list(reference_groups, multiple_reference_groups=True):
    colors_dictionary = {
        "random_guess": "#BEBEBE",
        "perfect_model": "#BEBEBE",
        "treat_none": "#BEBEBE",
        "treat_all": "#BEBEBE",
    }
    for group in reference_groups:
        for key in (
            group,
            f"random_guess_{group}",
            f"perfect_model_{group}",
            f"treat_none_{group}",
            f"treat_all_{group}",
        ):
            colors_dictionary[key] = "#000000"

    return {
        "reference_group_keys": reference_groups,
        "multiple_reference_groups": multiple_reference_groups,
        "colors_dictionary": colors_dictionary,
    }


def test_time_curve_custom_colors_propagate_to_groups_and_references():
    curve_list = _curve_list(["model_a", "model_b"])

    result = _apply_color_values_times(curve_list, ["#111111", "#222222"])

    for group, expected_color in (
        ("model_a", "#111111"),
        ("model_b", "#222222"),
    ):
        for key in (
            group,
            f"random_guess_{group}",
            f"perfect_model_{group}",
            f"treat_none_{group}",
            f"treat_all_{group}",
        ):
            assert result["colors_dictionary"][key] == expected_color

    assert result["colors_dictionary"]["random_guess"] == "#BEBEBE"
    assert result["colors_dictionary"]["perfect_model"] == "#BEBEBE"
    assert result["colors_dictionary"]["treat_none"] == "#BEBEBE"
    assert result["colors_dictionary"]["treat_all"] == "#BEBEBE"


def test_time_curve_single_model_keeps_existing_black_style():
    curve_list = _curve_list(["model"], multiple_reference_groups=False)

    result = _apply_color_values_times(curve_list, ["#123456"])

    assert result["colors_dictionary"]["model"] == "#000000"


def test_time_curve_custom_colors_require_one_per_reference_group():
    curve_list = _curve_list(["model_a", "model_b"])

    with pytest.raises(ValueError, match="one color per reference group"):
        _apply_color_values_times(curve_list, ["#111111"])

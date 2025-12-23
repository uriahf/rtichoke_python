def test_create_calibration_curve_times_mixed_inputs():
    from rtichoke.calibration.calibration import create_calibration_curve_times
    import polars as pl

    probs_dict = {
        "full": pl.Series(
            "pr_failure18", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        )
    }
    reals_series = pl.Series("cancer_cr", [0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    times_series = pl.Series(
        "ttcancer", [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    )

    try:
        create_calibration_curve_times(
            probs_dict,
            reals_series,
            times_series,
            fixed_time_horizons=[1.0, 2.0, 3.0, 4.0, 5.0],
            heuristics_sets=[
                {"censoring_heuristic": "excluded", "competing_heuristic": "excluded"}
            ],
        )
    except TypeError:
        assert False, (
            "create_calibration_curve_times raised a TypeError with mixed input types"
        )

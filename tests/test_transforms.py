"""Tests for transforms module helper functions and add_cutoff_strata."""

import numpy as np
import polars as pl
import pytest

from rtichoke.processing.transforms import (
    _compute_probability_quantile_bin_indices,
    add_cutoff_strata,
)


def test_compute_probability_quantile_bin_indices_basic():
    probs = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    by = 0.2
    bin_idx, q = _compute_probability_quantile_bin_indices(probs, by)

    assert q == 5
    assert len(bin_idx) == len(probs)
    assert bin_idx.min() >= 0
    assert bin_idx.max() < q


def test_add_cutoff_strata_unchanged():
    df = pl.DataFrame(
        {
            "reference_group": ["g1"] * 5,
            "probs": [0.0, 0.25, 0.5, 0.75, 1.0],
            "reals": [0, 1, 0, 1, 0],
        }
    )

    res_prob = add_cutoff_strata(df, by=0.2, stratified_by=("probability_threshold",))
    assert "strata_probability_threshold" in res_prob.columns

    res_ppcr = add_cutoff_strata(df, by=0.2, stratified_by=("ppcr",))
    assert "strata_ppcr" in res_ppcr.columns
    assert isinstance(res_ppcr["strata_ppcr"].dtype, pl.Enum)

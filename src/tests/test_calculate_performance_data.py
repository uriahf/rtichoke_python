import unittest
import numpy as np
import pandas as pd
from numpy.testing import assert_allclose
from rtichoke import Rtichoke

# UnitTests created with ChatGPT! :-)


class TestPreparePerformanceTable(unittest.TestCase):
    def setUp(self):
        probs = {"pop1": np.array([0.7, 0.8, 0.9, 0.4, 0.2, 0.6, 0.5])}
        reals = {"pop1": np.array([1, 1, 1, 0, 0, 1, 0])}
        self.r = Rtichoke(probs=probs, reals=reals, by=0.1)

    def test_performance_table_type_and_size(self):
        self.assertIsInstance(self.r.performance_table_pt, pd.DataFrame)
        self.assertIsInstance(self.r.performance_table_ppcr, pd.DataFrame)

        self.assertEqual(self.r.performance_table_pt.shape, (11, 15))
        self.assertEqual(self.r.performance_table_ppcr.shape, (11, 15))

    def test_performance_data_contains_expected_columns(self):
        result_cols_pt = self.r.performance_table_pt.columns
        result_cols_ppcr = self.r.performance_table_ppcr.columns

        expected_cols = [
            "Population",
            "probability_threshold",
            "ppcr",
            "predicted_positives",
            "TP",
            "FP",
            "FN",
            "TN",
            "Sensitivity",
            "Specificity",
            "FPR",
            "PPV",
            "NPV",
            "lift",
            "Net_benefit",
        ]
        self.assertCountEqual(result_cols_pt, expected_cols)
        self.assertCountEqual(result_cols_ppcr, expected_cols)

    def test_performance_data_contains_correct_thresholds(self):
        expected_thresholds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        result_thresholds_pt = self.r.performance_table_pt["probability_threshold"]
        result_thresholds_ppcr = self.r.performance_table_ppcr["ppcr"][-1::-1]

        assert_allclose(result_thresholds_pt, expected_thresholds)
        assert_allclose(result_thresholds_ppcr, expected_thresholds)

    def test_performance_data_contains_expected_population_name(self):
        expected_pop_name = ["pop1"]
        pt_table_pop_name = self.r.performance_table_pt["Population"].unique()
        ppcr_table_pop_name = self.r.performance_table_ppcr["Population"].unique()

        self.assertEqual(expected_pop_name, pt_table_pop_name)
        self.assertEqual(expected_pop_name, ppcr_table_pop_name)

    def test_performance_data_with_two_populations(self):
        probs = {
            "pop1": np.array([0.7, 0.8, 0.9, 0.4, 0.2, 0.6, 0.5]),
            "pop2": np.array([0.7, 0.8, 0.9, 0.4, 0.2, 0.6, 0.5]),
        }
        reals = {
            "pop1": np.array([1, 1, 1, 0, 0, 1, 0]),
            "pop2": np.array([1, 1, 1, 0, 0, 1, 0]),
        }
        r = Rtichoke(probs=probs, reals=reals, by=0.1)

        expected_pop_name = ["pop1", "pop2"]
        pt_table_pop_name = r.performance_table_pt["Population"].unique()
        ppcr_table_pop_name = r.performance_table_ppcr["Population"].unique()

        self.assertEqual(expected_pop_name, pt_table_pop_name)
        self.assertEqual(expected_pop_name, ppcr_table_pop_name)

        self.assertEqual(r.performance_table_pt.shape, (15, 22))
        self.assertEqual(r.performance_table_ppcr.shape, (15, 22))


if __name__ == "__main__":
    unittest.main()

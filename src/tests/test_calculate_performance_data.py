import unittest
import numpy as np
import pandas as pd
from numpy import assert_allclose
from rtichoke.performance_data._performance_data import (
    prepare_performance_data,
    prepare_performance_table,
)

# UnitTests created with ChatGPT! :-)


class TestPreparePerformanceTable(unittest.TestCase):
    def setUp(self):
        self.probs = np.array([0.7, 0.8, 0.9, 0.4, 0.2, 0.6, 0.5])
        self.reals = np.array([1, 1, 1, 0, 0, 1, 0])
        self.by = 0.1
        self.stratified_by = "probability_threshold"
        self.pop_name = "pop1"

    def test_prepare_performance_table_returns_dataframe(self):
        result = prepare_performance_table(self.probs, self.reals)
        self.assertIsInstance(result, pd.DataFrame)

    def test_prepare_performance_data_contains_expected_columns(self):
        result = prepare_performance_table(self.probs, self.reals)
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
        self.assertCountEqual(result.columns, expected_cols)

    def test_prepare_performance_data_contains_correct_prob_thresholds(self):
        result = prepare_performance_table(self.probs, self.reals, by=self.by)
        expected_prob_thresholds = np.array(
            [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        )
        assert_allclose(
            result["probability_threshold"].values, expected_prob_thresholds
        )

    def test_prepare_performance_data_contains_expected_population_name(self):
        result = prepare_performance_table(
            self.probs, self.reals, pop_name=self.pop_name
        )
        self.assertTrue((result["Population"] == self.pop_name).all())


class TestPreparePerformanceData(unittest.TestCase):
    def test_single_probas_single_labels(self):
        probas = np.array([0.2, 0.4, 0.6, 0.8])
        labels = np.array([0, 0, 1, 1])
        result = prepare_performance_data(probas, labels)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(
            len(result), 101
        )  # 101 rows: 100 thresholds + 1 row for overall metrics
        self.assertEqual(result["Population"].nunique(), 1)  # 1 population

    def test_multiple_probas_single_labels(self):
        probas1 = np.array([0.2, 0.4, 0.6, 0.8])
        probas2 = np.array([0.3, 0.5, 0.7, 0.9])
        labels = np.array([0, 0, 1, 1])
        result = prepare_performance_data({"pop1": probas1, "pop2": probas2}, labels)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 202)  # 2 populations, 101 thresholds each
        self.assertEqual(result["Population"].nunique(), 2)  # 2 populations

    def test_multiple_probas_multiple_labels(self):
        probas1 = np.array([0.2, 0.4, 0.6, 0.8])
        probas2 = np.array([0.3, 0.5, 0.7, 0.9])
        labels1 = np.array([0, 0, 1, 1])
        labels2 = np.array([1, 0, 0, 1])
        result = prepare_performance_data(
            {"pop1": probas1, "pop2": probas2}, {"pop1": labels1, "pop2": labels2}
        )
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(
            len(result), 202
        )  # 2 populations, 2 label sets, 101 thresholds each
        self.assertEqual(result["Population"].nunique(), 2)  # 2 populations

    def test_wrong_inputs(self):
        with self.assertRaises(Exception):
            prepare_performance_data({"pop1": [0.1, 0.2]}, [0, 1, 1])
        with self.assertRaises(Exception):
            prepare_performance_data(1, 2)


if __name__ == "__main__":
    unittest.main()

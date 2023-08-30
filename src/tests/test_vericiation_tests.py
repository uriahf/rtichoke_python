import unittest
import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal, assert_allclose
from pandas.testing import assert_frame_equal
from rtichoke.helpers.validations import *
from rtichoke import Rtichoke


class TestCheckProbs(unittest.TestCase):
    def test_valid_probs(self):
        try:
            check_probs(np.array([0.1, 0.3, 0.6, 0.9]))
        except Exception as e:
            self.fail(f"Should return None when OK: {e}")

    def test_invalid_probs_low(self):
        with self.assertRaises(Exception):
            check_probs(np.array([-0.1, 0.3, 0.6, 0.9]))

    def test_invalid_probs_high(self):
        with self.assertRaises(Exception):
            check_probs(np.array([0.1, 0.3, 1.1, 0.9]))


###


class TestCheckProbsVsReals(unittest.TestCase):
    def test_valid_probs_vs_reals(self):
        try:
            probs = np.array([0.1, 0.3, 0.6, 0.9])
            reals = np.array([0, 1, 0, 1])
            check_probs_vs_reals(probs, reals)
        except Exception as e:
            self.fail(f"Should return None when OK: {e}")

    def test_invalid_shape(self):
        probs = np.array([0.1, 0.3, 0.6, 0.9])
        reals = np.array([0, 1])
        with self.assertRaises(Exception):
            check_probs_vs_reals(probs, reals)

    def test_invalid_len(self):
        probs = np.array([0.1])
        reals = np.array([0])
        with self.assertRaises(Exception):
            check_probs_vs_reals(probs, reals)


class TestCheckReals(unittest.TestCase):
    def test_valid_reals(self):
        # Test a valid input
        try:
            reals = np.array([0, 1, 1, 0, 1])
            check_reals(reals)
        except Exception as e:
            self.fail(f"Should return None when OK: {e}")

    def test_reals_with_negative_values(self):
        # Test that an exception is raised when reals include negative values
        reals = np.array([-1, 0, 1])
        with self.assertRaises(Exception):
            check_reals(reals)

    def test_reals_with_a_single_value(self):
        # Test that an exception is raised when reals include only positive outcomes
        reals = np.array([1, 1, 1])
        with self.assertRaises(Exception):
            check_reals(reals)

    def test_reals_with_non_binary_values(self):
        # Test that an exception is raised when reals include non-binary values
        reals = np.array([0, 1, 2])
        with self.assertRaises(Exception):
            check_reals(reals)


class TestCheckBy(unittest.TestCase):
    def test_valid_by_value(self):
        r = Rtichoke(
            probs=np.array([0.1, 0.5, 0.9]), reals=np.array([0, 1, 0]), by=0.25
        )

    def test_by_not_a_float(self):
        # Test that an exception is raised when the input is not a float
        with self.assertRaises(Exception) as e:
            r = Rtichoke(
                probs=np.array([0.1, 0.5, 0.9]), reals=np.array([0, 1, 0]), by="0.25"
            )
        self.assertEqual(
            str(e.exception), "Argument `by` must be a float,  0 > `by` <= 0.5"
        )

    def test_by_out_of_range(self):
        # Test that an exception is raised when the input is out of range
        with self.assertRaises(Exception) as e:
            r = Rtichoke(
                probs=np.array([0.1, 0.5, 0.9]), reals=np.array([0, 1, 0]), by="0.6"
            )
        self.assertEqual(
            str(e.exception), "Argument `by` must be a float,  0 > `by` <= 0.5"
        )


class TestValidateInputs(unittest.TestCase):
    def test_validate_inputs(self):
        r = Rtichoke(probs=np.array([0.1, 0.5, 0.9]), reals=np.array([0, 1, 0]), by=0.2)

        # test for valid input
        probs = np.array([0.1, 0.5, 0.9])
        reals = np.array([0, 1, 0])
        self.assertIsNone(r.validate_inputs(probs, reals))

        # test for un-equal sized arrays
        with self.assertRaises(Exception) as e:
            r.validate_inputs(np.array([0.1, 0.5]), np.array([0, 1, 0]))
        self.assertEqual(
            str(e.exception),
            "Probs and reals shapes are inconsistent ((2,) and (3,))",
        )

        # test for invalid probs
        with self.assertRaises(Exception) as e:
            r.validate_inputs(np.array([-0.1, 0.5, 1.1]), np.array([0, 1, 0]))
        self.assertEqual(str(e.exception), "Probs must be within [0, 1]")

        # test for invalid reals
        with self.assertRaises(Exception) as e:
            r.validate_inputs(np.array([0.1, 0.5, 0.9]), np.array([0, 1, 2]))
        self.assertEqual(str(e.exception), "Reals must include only 0's and 1's")


if __name__ == "__main__":
    unittest.main()

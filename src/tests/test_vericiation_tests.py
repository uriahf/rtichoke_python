import unittest
import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal, assert_allclose
from pandas.testing import assert_frame_equal
from rtichoke.helpers.validations import *


class TestCheckProbs(unittest.TestCase):
    def test_valid_probs(self):
        self.assertTrue(check_probs(np.array([0.1, 0.3, 0.6, 0.9])))

    def test_invalid_probs_low(self):
        with self.assertRaises(Exception):
            check_probs(np.array([-0.1, 0.3, 0.6, 0.9]))

    def test_invalid_probs_high(self):
        with self.assertRaises(Exception):
            check_probs(np.array([0.1, 0.3, 1.1, 0.9]))


###


class TestCheckStratifiedByInput(unittest.TestCase):
    def test_valid_stratified_by_probability_threshold(self):
        self.assertTrue(check_stratified_by_input("probability_threshold"))

    def test_valid_stratified_by_ppcr(self):
        self.assertTrue(check_stratified_by_input("ppcr"))

    def test_invalid_stratified_by(self):
        with self.assertRaises(Exception):
            check_stratified_by_input("invalid_stratified_by")


###


class TestCheckProbsVsReals(unittest.TestCase):
    def test_valid_probs_vs_reals(self):
        probs = np.array([0.1, 0.3, 0.6, 0.9])
        reals = np.array([0, 1, 0, 1])
        self.assertTrue(check_probs_vs_reals(probs, reals))

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
        reals = np.array([0, 1, 1, 0, 1])
        self.assertTrue(check_reals(reals))

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
    def test_valid_by(self):
        # Test a valid input
        by = 0.25
        self.assertTrue(check_by(by))

    def test_by_not_a_float(self):
        # Test that an exception is raised when the input is not a float
        by = "0.25"
        with self.assertRaises(Exception):
            check_by(by)

    def test_by_out_of_range(self):
        # Test that an exception is raised when the input is out of range
        by = 0.6
        with self.assertRaises(Exception):
            check_by(by)

    def test_check_stratified_by_input(self):
        # test for valid input
        self.assertTrue(check_stratified_by_input("ppcr"))

        # test for invalid input
        with self.assertRaises(Exception):
            check_stratified_by_input("random")


class TestValidateInputs(unittest.TestCase):
    def test_validate_inputs(self):
        # test for valid input
        self.assertIsNone(
            validate_inputs(np.array([0.1, 0.5, 0.9]), np.array([0, 1, 0]), 0.1, "ppcr")
        )

        # test for invalid input
        with self.assertRaises(Exception):
            validate_inputs(np.array([0.1, 0.5]), np.array([0, 1, 0]), 0.1, "ppcr")
        with self.assertRaises(Exception):
            validate_inputs(
                np.array([-0.1, 0.5, 1.1]), np.array([0, 1, 0]), 0.1, "ppcr"
            )
        with self.assertRaises(Exception):
            validate_inputs(np.array([0.1, 0.5, 0.9]), np.array([0, 1]), 0.1, "ppcr")
        with self.assertRaises(Exception):
            validate_inputs(np.array([0.1, 0.5, 0.9]), np.array([0, 1, 0]), 0, "ppcr")
        with self.assertRaises(Exception):
            validate_inputs(np.array([0.1, 0.5, 0.9]), np.array([0, 1, 0]), 0.6)

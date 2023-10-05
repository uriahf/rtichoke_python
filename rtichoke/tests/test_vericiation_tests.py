"""Sub-module to test Rtichoke validations"""
import unittest
import numpy as np
from rtichoke.helpers.validations import check_probs, check_probs_vs_reals, check_reals
from rtichoke import rtichoke


class TestCheckProbs(unittest.TestCase):
    """UnitTest to test check_probs method"""

    def test_valid_probs(self: object) -> None:
        """Testing valid probs"""
        try:
            check_probs(np.array([0.1, 0.3, 0.6, 0.9]))
        except ValueError as e:
            self.fail(f"Should return None when OK: {e}")

    def test_invalid_probs_low(self: object) -> None:
        """Test check_probs with negative probability"""
        with self.assertRaises(ValueError):
            check_probs(np.array([-0.1, 0.3, 0.6, 0.9]))

    def test_invalid_probs_high(self: object) -> None:
        """Test check_probs with probability > 1.0"""
        with self.assertRaises(ValueError):
            check_probs(np.array([0.1, 0.3, 1.1, 0.9]))


###


class TestCheckProbsVsReals(unittest.TestCase):
    """UnitTest class to test check_probs_vs_reals"""

    def test_valid_probs_vs_reals(self: object) -> None:
        """Test check_probs_vs_reals: valid probs and reals"""
        try:
            probs = np.array([0.1, 0.3, 0.6, 0.9])
            reals = np.array([0, 1, 0, 1])
            check_probs_vs_reals(probs, reals)
        except ValueError as e:
            self.fail(f"Should return None when OK: {e}")

    def test_invalid_shape(self: object) -> None:
        """Test check_probs_vs_reals: shape mis-match"""
        probs = np.array([0.1, 0.3, 0.6, 0.9])
        reals = np.array([0, 1])
        with self.assertRaises(ValueError):
            check_probs_vs_reals(probs, reals)

    def test_invalid_len(self: object) -> None:
        """Test check_probs_vs_reals: single observation"""
        probs = np.array([0.1])
        reals = np.array([0])
        with self.assertRaises(ValueError):
            check_probs_vs_reals(probs, reals)


class TestCheckReals(unittest.TestCase):
    """UnitTest class to test check_reals"""

    def test_valid_reals(self: object) -> None:
        """Test check_reals: valid input"""
        # Test a valid input
        try:
            reals = np.array([0, 1, 1, 0, 1])
            check_reals(reals)
        except ValueError as e:
            self.fail(f"Should return None when OK: {e}")

    def test_reals_with_negative_values(self: object) -> None:
        """Test check_reals: negative reals"""
        # Test that an exception is raised when reals include negative values
        reals = np.array([-1, 0, 1])
        with self.assertRaises(ValueError):
            check_reals(reals)

    def test_reals_with_a_single_value(self: object) -> None:
        """Test check_reals: a single unique reals value"""
        # Test that an exception is raised when reals include only positive outcomes
        reals = np.array([1, 1, 1])
        with self.assertRaises(ValueError):
            check_reals(reals)

    def test_reals_with_non_binary_values(self: object) -> None:
        """Test check_reals: multi-nomial reals"""
        # Test that an exception is raised when reals include non-binary values
        reals = np.array([0, 1, 2])
        with self.assertRaises(ValueError):
            check_reals(reals)


class TestCheckBy(unittest.TestCase):
    """UnitTest class to test check_by"""

    def test_valid_by_value(self: object) -> None:
        """Test check_by: valid value"""
        rtichoke.Rtichoke(
            probs=np.array([0.1, 0.5, 0.9]), reals=np.array([0, 1, 0]), by=0.25
        )

    def test_by_not_a_float(self: object) -> None:
        """Test check_by: `by` argument not a float"""
        # Test that an exception is raised when the input is not a float
        with self.assertRaises(ValueError) as e:
            rtichoke.Rtichoke(
                probs=np.array([0.1, 0.5, 0.9]), reals=np.array([0, 1, 0]), by="0.25"
            )
        self.assertEqual(
            str(e.exception), "Argument `by` must be a float,  0 > `by` <= 0.5"
        )

    def test_by_out_of_range(self: object) -> None:
        """Test check_by: `by` argument out of range"""
        # Test that an exception is raised when the input is out of range
        with self.assertRaises(ValueError) as e:
            rtichoke.Rtichoke(
                probs=np.array([0.1, 0.5, 0.9]), reals=np.array([0, 1, 0]), by="0.6"
            )
        self.assertEqual(
            str(e.exception), "Argument `by` must be a float,  0 > `by` <= 0.5"
        )


class TestValidateInputs(unittest.TestCase):
    """UnitTest class for end-to-end validation tests"""

    def test_validate_inputs(self: object) -> None:
        """End-to-end test with valid inputs"""
        r = rtichoke.Rtichoke(
            probs=np.array([0.1, 0.5, 0.9]), reals=np.array([0, 1, 0]), by=0.2
        )

        # test for valid input
        probs = np.array([0.1, 0.5, 0.9])
        reals = np.array([0, 1, 0])
        self.assertIsNone(r.validate_inputs(probs, reals))

        # test for un-equal sized arrays
        with self.assertRaises(ValueError) as e:
            r.validate_inputs(np.array([0.1, 0.5]), np.array([0, 1, 0]))
        self.assertEqual(
            str(e.exception),
            "Probs and reals shapes are inconsistent ((2,) and (3,))",
        )

        # test for invalid probs
        with self.assertRaises(ValueError) as e:
            r.validate_inputs(np.array([-0.1, 0.5, 1.1]), np.array([0, 1, 0]))
        self.assertEqual(str(e.exception), "Probs must be within [0, 1]")

        # test for invalid reals
        with self.assertRaises(ValueError) as e:
            r.validate_inputs(np.array([0.1, 0.5, 0.9]), np.array([0, 1, 2]))
        self.assertEqual(str(e.exception), "Reals must include only 0's and 1's")


if __name__ == "__main__":
    unittest.main()

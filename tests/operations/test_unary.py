"""Tests for unary operations dispatched from polder top-level API."""

import numpy as np
import polars as pl
import pytest

import polder as pld


@pytest.mark.parametrize(
    "func,numpy_func,test_values",
    [
        # Trigonometric functions
        (pld.acos, np.acos, np.array([[-0.5, 0.0], [0.5, 1.0]])),
        (pld.asin, np.asin, np.array([[-0.5, 0.0], [0.5, 1.0]])),
        (pld.atan, np.atan, np.array([[-1.0, 0.0], [1.0, 2.0]])),
        (pld.cos, np.cos, np.array([[0.0, np.pi / 4], [np.pi / 2, np.pi]])),
        (pld.sin, np.sin, np.array([[0.0, np.pi / 4], [np.pi / 2, np.pi]])),
        (pld.tan, np.tan, np.array([[0.0, np.pi / 4], [0.1, 0.2]])),
        # Inverse hyperbolic functions
        (pld.acosh, np.acosh, np.array([[1.0, 2.0], [3.0, 4.0]])),
        (pld.asinh, np.asinh, np.array([[-2.0, -1.0], [1.0, 2.0]])),
        (pld.atanh, np.atanh, np.array([[-0.5, 0.0], [0.25, 0.5]])),
        # Hyperbolic functions
        (pld.cosh, np.cosh, np.array([[0.0, 1.0], [2.0, 3.0]])),
        (pld.sinh, np.sinh, np.array([[0.0, 1.0], [2.0, 3.0]])),
        (pld.tanh, np.tanh, np.array([[0.0, 1.0], [2.0, 3.0]])),
        # Exponential and logarithmic
        (pld.exp, np.exp, np.array([[0.0, 1.0], [2.0, 0.5]])),
        (pld.expm1, np.expm1, np.array([[0.0, 1.0], [2.0, 0.5]])),
        (pld.log, np.log, np.array([[1.0, 2.0], [3.0, 4.0]])),
        (pld.log1p, np.log1p, np.array([[0.0, 1.0], [2.0, 3.0]])),
        (pld.log2, np.log2, np.array([[1.0, 2.0], [4.0, 8.0]])),
        (pld.log10, np.log10, np.array([[1.0, 10.0], [100.0, 1000.0]])),
        # Rounding
        (pld.ceil, np.ceil, np.array([[1.2, -1.5], [2.7, -0.3]])),
        (pld.floor, np.floor, np.array([[1.2, -1.5], [2.7, -0.3]])),
        (pld.round_, np.round, np.array([[1.2, 1.5], [2.7, -0.3]])),
        (pld.trunc, np.trunc, np.array([[1.2, -1.5], [2.7, -0.3]])),
        # Sign and reciprocal
        (pld.reciprocal, np.reciprocal, np.array([[0.5, 1.0], [2.0, 4.0]])),
        (pld.sign, np.sign, np.array([[-2.0, -0.5], [0.5, 2.0]])),
        (pld.sqrt, np.sqrt, np.array([[1.0, 4.0], [9.0, 16.0]])),
        (pld.square, np.square, np.array([[1.0, -2.0], [3.0, -4.0]])),
        # Bitwise
        (pld.bitwise_invert, np.bitwise_invert, np.array([[1.0, 2.0], [3.0, 4.0]])),
        # Logical
        (pld.logical_not, np.logical_not, np.array([[True, False], [False, True]])),
    ],
)
def test_unary_elementwise_functions(func, numpy_func, test_values):
    """Test unary elementwise functions against numpy equivalents."""
    # Cast to appropriate dtype for bitwise operations.
    if func is pld.bitwise_invert:
        test_values = test_values.astype(np.int32)

    labels = [
        pl.DataFrame({"x": [0, 1]}),
        pl.DataFrame({"y": [0, 1]}),
    ]
    arr = pld.from_values_and_labels(test_values, labels)

    result = func(arr)

    # Compare to numpy.
    expected_values = numpy_func(test_values)
    np.testing.assert_array_almost_equal(result.values(), expected_values)

    # Labels should be preserved.
    assert result.labels() == arr.labels()


@pytest.mark.parametrize(
    "func,numpy_func",
    [
        (pld.imag, np.imag),
        (pld.real, np.real),
        (pld.conj, np.conj),
    ],
)
def test_unary_complex_functions(func, numpy_func):
    """Test complex-valued unary functions."""
    test_values = np.array([[1 + 2j, 3 - 4j], [5 + 0j, 0 - 6j]])
    labels = [
        pl.DataFrame({"x": [0, 1]}),
        pl.DataFrame({"y": [0, 1]}),
    ]
    arr = pld.from_values_and_labels(test_values, labels)

    result = func(arr)

    expected_values = numpy_func(test_values)
    np.testing.assert_array_equal(result.values(), expected_values)

    assert result.labels() == arr.labels()


@pytest.mark.parametrize(
    "func,numpy_func",
    [
        (pld.isfinite, np.isfinite),
        (pld.isinf, np.isinf),
        (pld.isnan, np.isnan),
        (pld.signbit, np.signbit),
    ],
)
def test_unary_classification_functions(func, numpy_func):
    """Test classification functions that return boolean or int."""
    test_values = np.array([[np.inf, -np.inf], [np.nan, 1.0]])
    labels = [
        pl.DataFrame({"x": [0, 1]}),
        pl.DataFrame({"y": [0, 1]}),
    ]
    arr = pld.from_values_and_labels(test_values, labels)

    result = func(arr)

    expected_values = numpy_func(test_values)
    np.testing.assert_array_equal(result.values(), expected_values)

    assert result.labels() == arr.labels()

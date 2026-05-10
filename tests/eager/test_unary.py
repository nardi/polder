import numpy as np
import polars as pl

import polder as pld

from ..utils import cast_value_array


def test_pos(value_array_type):
    """Test unary positive operator."""
    values = cast_value_array(np.array([[-1.0, 2.0], [3.0, -4.0]]), value_array_type)
    labels = [
        pl.DataFrame({"x": [0, 1]}),
        pl.DataFrame({"y": [0, 1]}),
    ]
    arr = pld.from_values_and_labels(values, labels)

    result = +arr
    assert isinstance(result.values(), value_array_type.value)

    # Values should be the same (pos is identity).
    np.testing.assert_array_equal(result.values(), values)

    # Labels should be preserved.
    assert result.equals(arr)


def test_neg(value_array_type):
    """Test unary negation operator."""
    values = cast_value_array(np.array([[1.0, -2.0], [-3.0, 4.0]]), value_array_type)
    labels = [
        pl.DataFrame({"x": [0, 1]}),
        pl.DataFrame({"y": [0, 1]}),
    ]
    arr = pld.from_values_and_labels(values, labels)

    result = -arr
    assert isinstance(result.values(), value_array_type.value)

    # Values should be negated.
    expected_values = np.array([[-1.0, 2.0], [3.0, -4.0]])
    np.testing.assert_array_equal(result.values(), expected_values)

    # Labels should be preserved.
    assert result.labels() == arr.labels()


def test_abs(value_array_type):
    """Test absolute value operator."""
    values = cast_value_array(np.array([[1.0, -2.5], [-3.7, 4.2]]), value_array_type)
    labels = [
        pl.DataFrame({"x": [0, 1]}),
        pl.DataFrame({"y": [0, 1]}),
    ]
    arr = pld.from_values_and_labels(values, labels)

    result = abs(arr)
    assert isinstance(result.values(), value_array_type.value)

    # Values should be absolute.
    expected_values = np.array([[1.0, 2.5], [3.7, 4.2]])
    np.testing.assert_array_almost_equal(result.values(), expected_values)

    # Labels should be preserved.
    assert result.labels() == arr.labels()


def test_invert(value_array_type):
    """Test bitwise invert operator."""
    values = cast_value_array(
        np.array([[1, 2], [3, 4]], dtype=np.int32), value_array_type
    )
    labels = [
        pl.DataFrame({"x": [0, 1]}),
        pl.DataFrame({"y": [0, 1]}),
    ]
    arr = pld.from_values_and_labels(values, labels)

    result = ~arr
    assert isinstance(result.values(), value_array_type.value)

    # Values should be bitwise inverted.
    expected_values = ~values
    np.testing.assert_array_equal(result.values(), expected_values)

    # Labels should be preserved.
    assert result.labels() == arr.labels()


def test_invert_bool(value_array_type):
    """Test bitwise invert operator with boolean arrays."""
    values = cast_value_array(
        np.array([[True, False], [False, True]]), value_array_type
    )
    labels = [
        pl.DataFrame({"x": [0, 1]}),
        pl.DataFrame({"y": [0, 1]}),
    ]
    arr = pld.from_values_and_labels(values, labels)

    result = ~arr
    assert isinstance(result.values(), value_array_type.value)

    # Values should be inverted.
    expected_values = np.array([[False, True], [True, False]])
    np.testing.assert_array_equal(result.values(), expected_values)

    # Labels should be preserved.
    assert result.labels() == arr.labels()


def test_neg_complex(value_array_type):
    """Test negation with complex numbers."""
    values = cast_value_array(
        np.array([[1 + 2j, 3 - 4j], [-5 + 0j, 0 - 6j]]), value_array_type
    )
    labels = [
        pl.DataFrame({"x": [0, 1]}),
        pl.DataFrame({"y": [0, 1]}),
    ]
    arr = pld.from_values_and_labels(values, labels)

    result = -arr
    assert isinstance(result.values(), value_array_type.value)

    # Values should be negated.
    expected_values = -values
    np.testing.assert_array_equal(result.values(), expected_values)

    # Labels should be preserved.
    assert result.labels() == arr.labels()


def test_abs_complex(value_array_type):
    """Test absolute value with complex numbers."""
    values = cast_value_array(
        np.array([[3 + 4j, 1 + 1j], [0 + 2j, 1 + 0j]]), value_array_type
    )
    labels = [
        pl.DataFrame({"x": [0, 1]}),
        pl.DataFrame({"y": [0, 1]}),
    ]
    arr = pld.from_values_and_labels(values, labels)

    result = abs(arr)
    assert isinstance(result.values(), value_array_type.value)

    # Values should be absolute (magnitude).
    expected_values = np.abs(values)
    np.testing.assert_array_almost_equal(result.values(), expected_values)

    # Labels should be preserved.
    assert result.labels() == arr.labels()


def test_unary_preserves_shape(value_array_type):
    """Test that unary operators preserve shape."""
    values = cast_value_array(
        np.arange(2 * 3 * 4).reshape(2, 3, 4).astype(float), value_array_type
    )
    labels = [
        pl.DataFrame({"a": [0, 1]}),
        pl.DataFrame({"b": [0, 1, 2]}),
        pl.DataFrame({"c": [0, 1, 2, 3]}),
    ]
    arr = pld.from_values_and_labels(values, labels)

    # Test all unary operators preserve shape.
    assert (+arr).shape() == arr.shape()
    assert (-arr).shape() == arr.shape()
    assert abs(arr).shape() == arr.shape()


def test_chained_unary_operators(value_array_type):
    """Test chaining multiple unary operators."""
    values = cast_value_array(np.array([[-1.0, -2.0], [3.0, -4.0]]), value_array_type)
    labels = [
        pl.DataFrame({"x": [0, 1]}),
        pl.DataFrame({"y": [0, 1]}),
    ]
    arr = pld.from_values_and_labels(values, labels)

    # Double negation should return to original.
    result = -(-arr)
    assert isinstance(result.values(), value_array_type.value)
    np.testing.assert_array_equal(result.values(), values)

    # Abs then negate.
    result = -(abs(arr))
    assert isinstance(result.values(), value_array_type.value)
    expected_values = -np.abs(values)
    np.testing.assert_array_equal(result.values(), expected_values)

    # Labels should be preserved through all operations.
    assert result.labels() == arr.labels()


def test_unary_with_nan(value_array_type):
    """Test unary operators with NaN values."""
    values = cast_value_array(
        np.array([[np.nan, 1.0], [2.0, np.nan]]), value_array_type
    )
    labels = [
        pl.DataFrame({"x": [0, 1]}),
        pl.DataFrame({"y": [0, 1]}),
    ]
    arr = pld.from_values_and_labels(values, labels)

    result = -arr
    assert isinstance(result.values(), value_array_type.value)

    # NaN should remain NaN.
    assert np.isnan(result.values(0, 0))
    assert result.values(0, 1) == -1.0
    assert result.values(1, 0) == -2.0
    assert np.isnan(result.values(1, 1))

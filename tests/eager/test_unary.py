import numpy as np
import polars as pl

import polder as pld


def test_pos():
    """Test unary positive operator."""
    values = np.array([[-1.0, 2.0], [3.0, -4.0]])
    labels = [
        pl.DataFrame({"x": [0, 1]}),
        pl.DataFrame({"y": [0, 1]}),
    ]
    arr = pld.from_values_and_labels(values, labels)

    result = +arr

    # Values should be the same (pos is identity).
    np.testing.assert_array_equal(result.values(), values)

    # Labels should be preserved.
    assert result.equals(arr)


def test_neg():
    """Test unary negation operator."""
    values = np.array([[1.0, -2.0], [-3.0, 4.0]])
    labels = [
        pl.DataFrame({"x": [0, 1]}),
        pl.DataFrame({"y": [0, 1]}),
    ]
    arr = pld.from_values_and_labels(values, labels)

    result = -arr

    # Values should be negated.
    expected_values = np.array([[-1.0, 2.0], [3.0, -4.0]])
    np.testing.assert_array_equal(result.values(), expected_values)

    # Labels should be preserved.
    assert result.labels() == arr.labels()


def test_abs():
    """Test absolute value operator."""
    values = np.array([[1.0, -2.5], [-3.7, 4.2]])
    labels = [
        pl.DataFrame({"x": [0, 1]}),
        pl.DataFrame({"y": [0, 1]}),
    ]
    arr = pld.from_values_and_labels(values, labels)

    result = abs(arr)

    # Values should be absolute.
    expected_values = np.array([[1.0, 2.5], [3.7, 4.2]])
    np.testing.assert_array_equal(result.values(), expected_values)

    # Labels should be preserved.
    assert result.labels() == arr.labels()


def test_invert():
    """Test bitwise invert operator."""
    values = np.array([[1, 2], [3, 4]], dtype=np.int32)
    labels = [
        pl.DataFrame({"x": [0, 1]}),
        pl.DataFrame({"y": [0, 1]}),
    ]
    arr = pld.from_values_and_labels(values, labels)

    result = ~arr

    # Values should be bitwise inverted.
    expected_values = ~values
    np.testing.assert_array_equal(result.values(), expected_values)

    # Labels should be preserved.
    assert result.labels() == arr.labels()


def test_invert_bool():
    """Test bitwise invert operator with boolean arrays."""
    values = np.array([[True, False], [False, True]])
    labels = [
        pl.DataFrame({"x": [0, 1]}),
        pl.DataFrame({"y": [0, 1]}),
    ]
    arr = pld.from_values_and_labels(values, labels)

    result = ~arr

    # Values should be inverted.
    expected_values = np.array([[False, True], [True, False]])
    np.testing.assert_array_equal(result.values(), expected_values)

    # Labels should be preserved.
    assert result.labels() == arr.labels()


def test_neg_complex():
    """Test negation with complex numbers."""
    values = np.array([[1+2j, 3-4j], [-5+0j, 0-6j]])
    labels = [
        pl.DataFrame({"x": [0, 1]}),
        pl.DataFrame({"y": [0, 1]}),
    ]
    arr = pld.from_values_and_labels(values, labels)

    result = -arr

    # Values should be negated.
    expected_values = -values
    np.testing.assert_array_equal(result.values(), expected_values)

    # Labels should be preserved.
    assert result.labels() == arr.labels()


def test_abs_complex():
    """Test absolute value with complex numbers."""
    values = np.array([[3+4j, 1+1j], [0+2j, 1+0j]])
    labels = [
        pl.DataFrame({"x": [0, 1]}),
        pl.DataFrame({"y": [0, 1]}),
    ]
    arr = pld.from_values_and_labels(values, labels)

    result = abs(arr)

    # Values should be absolute (magnitude).
    expected_values = np.abs(values)
    np.testing.assert_array_almost_equal(result.values(), expected_values)

    # Labels should be preserved.
    assert result.labels() == arr.labels()


def test_unary_preserves_shape():
    """Test that unary operators preserve shape."""
    values = np.arange(2 * 3 * 4).reshape(2, 3, 4).astype(float)
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


def test_chained_unary_operators():
    """Test chaining multiple unary operators."""
    values = np.array([[-1.0, -2.0], [3.0, -4.0]])
    labels = [
        pl.DataFrame({"x": [0, 1]}),
        pl.DataFrame({"y": [0, 1]}),
    ]
    arr = pld.from_values_and_labels(values, labels)

    # Double negation should return to original.
    result = -(-arr)
    np.testing.assert_array_equal(result.values(), values)

    # Abs then negate.
    result = -(abs(arr))
    expected_values = -np.abs(values)
    np.testing.assert_array_equal(result.values(), expected_values)

    # Labels should be preserved through all operations.
    assert result.labels() == arr.labels()


def test_unary_with_nan():
    """Test unary operators with NaN values."""
    values = np.array([[np.nan, 1.0], [2.0, np.nan]])
    labels = [
        pl.DataFrame({"x": [0, 1]}),
        pl.DataFrame({"y": [0, 1]}),
    ]
    arr = pld.from_values_and_labels(values, labels)

    result = -arr

    # NaN should remain NaN.
    assert np.isnan(result.values()[0, 0])
    assert result.values()[0, 1] == -1.0
    assert result.values()[1, 0] == -2.0
    assert np.isnan(result.values()[1, 1])

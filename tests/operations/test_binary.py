"""Tests for binary operations dispatched from polder.operations, parametrized over eager/lazy."""

from typing import Any, Sequence

import narwhals as nw
import numpy as np
import polars as pl
import pytest

import polder as pld
from polder.eager.labels import Labels
from polder.protocols.array import FrameLabeledArray
from polder.protocols.implementations import (
    EAGER,
    LAZY,
    FrameLabeledArrayImplementation,
)


@pytest.fixture(params=[EAGER, LAZY])
def implementation(request):
    """Parametrized fixture for testing both eager and lazy implementations."""
    return request.param


def create_array(
    values: np.ndarray,
    labels: Sequence[pl.DataFrame | None],
    implementation: FrameLabeledArrayImplementation,
):
    """Create either an eager or lazy array depending on implementation."""
    return pld.from_values_and_labels(values, labels, implementation=implementation)


def assert_array_labels_equal(
    result: FrameLabeledArray, expected: Sequence[nw.DataFrame[Any] | None]
) -> None:
    """Assert that array labels match expected (handling lazy evaluation)."""
    assert Labels(result.labels()) == Labels(expected)


def assert_array_values_equal(result: FrameLabeledArray, expected: np.ndarray) -> None:
    """Assert that array values match expected (handling lazy evaluation)."""
    np.testing.assert_array_almost_equal(result.values(), expected)


def assert_array_shape_equal(
    result: FrameLabeledArray, expected: tuple[int, ...]
) -> None:
    """Assert that array shape matches expected."""
    assert result.shape() == expected


# ============================================================================
# Tests for arithmetic operations
# ============================================================================


def test_add_arrays(implementation: FrameLabeledArrayImplementation) -> None:
    """Test adding two arrays."""
    values1 = np.array([[1.0, 2.0], [3.0, 4.0]])
    values2 = np.array([[5.0, 6.0], [7.0, 8.0]])
    labels = [pl.DataFrame({"x": [0, 1]}), pl.DataFrame({"y": [0, 1]})]

    array1 = create_array(values1, labels, implementation)
    array2 = create_array(values2, labels, implementation)

    result = array1 + array2  # type: ignore

    assert_array_shape_equal(result, (2, 2))
    assert_array_values_equal(result, values1 + values2)


def test_add_scalar_to_array(implementation: FrameLabeledArrayImplementation) -> None:
    """Test adding a scalar to an array."""
    values = np.array([[1.0, 2.0], [3.0, 4.0]])
    labels = [pl.DataFrame({"x": [0, 1]}), pl.DataFrame({"y": [0, 1]})]
    array = create_array(values, labels, implementation)
    scalar = 5.0

    result = array + scalar

    assert_array_shape_equal(result, (2, 2))
    assert_array_values_equal(result, values + scalar)


def test_add_array_to_scalar(implementation: FrameLabeledArrayImplementation) -> None:
    """Test adding an array to a scalar (reflected operation)."""
    values = np.array([[1.0, 2.0], [3.0, 4.0]])
    labels = [pl.DataFrame({"x": [0, 1]}), pl.DataFrame({"y": [0, 1]})]
    array = create_array(values, labels, implementation)
    scalar = 3.0

    result = scalar + array

    assert_array_shape_equal(result, (2, 2))
    assert_array_values_equal(result, scalar + values)


def test_subtract_arrays(implementation: FrameLabeledArrayImplementation) -> None:
    """Test subtracting arrays."""
    values1 = np.array([[1.0, 2.0], [3.0, 4.0]])
    values2 = np.array([[5.0, 6.0], [7.0, 8.0]])
    labels = [pl.DataFrame({"x": [0, 1]}), pl.DataFrame({"y": [0, 1]})]

    array1 = create_array(values1, labels, implementation)
    array2 = create_array(values2, labels, implementation)

    result = array1 - array2  # type: ignore

    assert_array_shape_equal(result, (2, 2))
    assert_array_values_equal(result, values1 - values2)


def test_subtract_scalar_from_array(
    implementation: FrameLabeledArrayImplementation,
) -> None:
    """Test subtracting a scalar from an array."""
    values = np.array([[1.0, 2.0], [3.0, 4.0]])
    labels = [pl.DataFrame({"x": [0, 1]}), pl.DataFrame({"y": [0, 1]})]
    array = create_array(values, labels, implementation)
    scalar = 2.0

    result = array - scalar

    assert_array_shape_equal(result, (2, 2))
    assert_array_values_equal(result, values - scalar)


def test_subtract_with_reflected(
    implementation: FrameLabeledArrayImplementation,
) -> None:
    """Test subtraction with reflected operation (scalar - array)."""
    values = np.array([[1.0, 2.0], [3.0, 4.0]])
    labels = [pl.DataFrame({"x": [0, 1]}), pl.DataFrame({"y": [0, 1]})]
    array = create_array(values, labels, implementation)
    scalar = 2.0

    result = scalar - array

    assert_array_shape_equal(result, (2, 2))
    assert_array_values_equal(result, scalar - values)


def test_multiply_arrays(implementation: FrameLabeledArrayImplementation) -> None:
    """Test multiplying arrays."""
    values1 = np.array([[1.0, 2.0], [3.0, 4.0]])
    values2 = np.array([[5.0, 6.0], [7.0, 8.0]])
    labels = [pl.DataFrame({"x": [0, 1]}), pl.DataFrame({"y": [0, 1]})]

    array1 = create_array(values1, labels, implementation)
    array2 = create_array(values2, labels, implementation)

    result = array1 * array2  # type: ignore

    assert_array_shape_equal(result, (2, 2))
    assert_array_values_equal(result, values1 * values2)


def test_multiply_scalar_to_array(
    implementation: FrameLabeledArrayImplementation,
) -> None:
    """Test multiplying an array by a scalar."""
    values = np.array([[1.0, 2.0], [3.0, 4.0]])
    labels = [pl.DataFrame({"x": [0, 1]}), pl.DataFrame({"y": [0, 1]})]
    array = create_array(values, labels, implementation)
    scalar = 4.0

    result = array * scalar

    assert_array_shape_equal(result, (2, 2))
    assert_array_values_equal(result, values * scalar)


def test_multiply_array_by_scalar(
    implementation: FrameLabeledArrayImplementation,
) -> None:
    """Test multiplying a scalar by an array (reflected operation)."""
    values = np.array([[1.0, 2.0], [3.0, 4.0]])
    labels = [pl.DataFrame({"x": [0, 1]}), pl.DataFrame({"y": [0, 1]})]
    array = create_array(values, labels, implementation)
    scalar = 4.0

    result = scalar * array

    assert_array_shape_equal(result, (2, 2))
    assert_array_values_equal(result, scalar * values)


def test_divide_arrays(implementation: FrameLabeledArrayImplementation) -> None:
    """Test dividing arrays."""
    values1 = np.array([[2.0, 4.0], [6.0, 8.0]])
    values2 = np.array([[1.0, 2.0], [3.0, 4.0]])
    labels = [pl.DataFrame({"x": [0, 1]}), pl.DataFrame({"y": [0, 1]})]

    array1 = create_array(values1, labels, implementation)
    array2 = create_array(values2, labels, implementation)

    result = array1 / array2  # type: ignore

    assert_array_shape_equal(result, (2, 2))
    assert_array_values_equal(result, values1 / values2)


def test_divide_array_by_scalar(
    implementation: FrameLabeledArrayImplementation,
) -> None:
    """Test dividing an array by a scalar."""
    values = np.array([[2.0, 4.0], [6.0, 8.0]])
    labels = [pl.DataFrame({"x": [0, 1]}), pl.DataFrame({"y": [0, 1]})]
    array = create_array(values, labels, implementation)
    scalar = 2.0

    result = array / scalar

    assert_array_shape_equal(result, (2, 2))
    assert_array_values_equal(result, values / scalar)


def test_divide_with_reflected(implementation: FrameLabeledArrayImplementation) -> None:
    """Test division with reflected operation (scalar / array)."""
    values = np.array([[1.0, 2.0], [4.0, 5.0]])
    labels = [pl.DataFrame({"x": [0, 1]}), pl.DataFrame({"y": [0, 1]})]
    array = create_array(values, labels, implementation)
    scalar = 4.0

    result = scalar / array

    assert_array_shape_equal(result, (2, 2))
    assert_array_values_equal(result, scalar / values)


def test_floor_divide_arrays(implementation: FrameLabeledArrayImplementation) -> None:
    """Test floor dividing arrays."""
    values1 = np.array([[5.0, 6.0], [7.0, 8.0]])
    values2 = np.array([[2.0, 2.0], [2.0, 2.0]])
    labels = [pl.DataFrame({"x": [0, 1]}), pl.DataFrame({"y": [0, 1]})]

    array1 = create_array(values1, labels, implementation)
    array2 = create_array(values2, labels, implementation)

    result = array1 // array2  # type: ignore

    assert_array_shape_equal(result, (2, 2))
    assert_array_values_equal(result, np.floor(values1 / values2))


def test_modulo_arrays(implementation: FrameLabeledArrayImplementation) -> None:
    """Test modulo operation on arrays."""
    values1 = np.array([[5.0, 6.0], [7.0, 8.0]])
    values2 = np.array([[2.0, 3.0], [3.0, 3.0]])
    labels = [pl.DataFrame({"x": [0, 1]}), pl.DataFrame({"y": [0, 1]})]

    array1 = create_array(values1, labels, implementation)
    array2 = create_array(values2, labels, implementation)

    result = array1 % array2  # type: ignore

    assert_array_shape_equal(result, (2, 2))
    assert_array_values_equal(result, values1 % values2)


def test_power_arrays(implementation: FrameLabeledArrayImplementation) -> None:
    """Test power operation on arrays."""
    values1 = np.array([[2.0, 3.0], [2.0, 3.0]])
    values2 = np.array([[1.0, 2.0], [3.0, 1.0]])
    labels = [pl.DataFrame({"x": [0, 1]}), pl.DataFrame({"y": [0, 1]})]

    array1 = create_array(values1, labels, implementation)
    array2 = create_array(values2, labels, implementation)

    result = array1**array2  # type: ignore

    assert_array_shape_equal(result, (2, 2))
    assert_array_values_equal(result, values1**values2)


# ============================================================================
# Tests for bitwise operations
# ============================================================================


def test_bitwise_and(implementation: FrameLabeledArrayImplementation) -> None:
    """Test bitwise AND operation."""
    values1 = np.array([[5, 6], [7, 8]], dtype=int)
    values2 = np.array([[3, 4], [2, 1]], dtype=int)
    labels = [pl.DataFrame({"x": [0, 1]}), pl.DataFrame({"y": [0, 1]})]

    array1 = create_array(values1, labels, implementation)
    array2 = create_array(values2, labels, implementation)

    result = array1 & array2  # type: ignore

    assert_array_shape_equal(result, (2, 2))
    assert_array_values_equal(result, values1 & values2)


def test_bitwise_or(implementation: FrameLabeledArrayImplementation) -> None:
    """Test bitwise OR operation."""
    values1 = np.array([[5, 6], [7, 8]], dtype=int)
    values2 = np.array([[3, 4], [2, 1]], dtype=int)
    labels = [pl.DataFrame({"x": [0, 1]}), pl.DataFrame({"y": [0, 1]})]

    array1 = create_array(values1, labels, implementation)
    array2 = create_array(values2, labels, implementation)

    result = array1 | array2  # type: ignore

    assert_array_shape_equal(result, (2, 2))
    assert_array_values_equal(result, values1 | values2)


def test_bitwise_xor(implementation: FrameLabeledArrayImplementation) -> None:
    """Test bitwise XOR operation."""
    values1 = np.array([[5, 6], [7, 8]], dtype=int)
    values2 = np.array([[3, 4], [2, 1]], dtype=int)
    labels = [pl.DataFrame({"x": [0, 1]}), pl.DataFrame({"y": [0, 1]})]

    array1 = create_array(values1, labels, implementation)
    array2 = create_array(values2, labels, implementation)

    result = array1 ^ array2  # type: ignore

    assert_array_shape_equal(result, (2, 2))
    assert_array_values_equal(result, values1 ^ values2)


def test_bitwise_left_shift(implementation: FrameLabeledArrayImplementation) -> None:
    """Test bitwise left shift operation."""
    values1 = np.array([[1, 2], [3, 4]], dtype=int)
    values2 = np.array([[1, 2], [1, 2]], dtype=int)
    labels = [pl.DataFrame({"x": [0, 1]}), pl.DataFrame({"y": [0, 1]})]

    array1 = create_array(values1, labels, implementation)
    array2 = create_array(values2, labels, implementation)

    result = array1 << array2  # type: ignore

    assert_array_shape_equal(result, (2, 2))
    assert_array_values_equal(result, values1 << values2)


def test_bitwise_right_shift(implementation: FrameLabeledArrayImplementation) -> None:
    """Test bitwise right shift operation."""
    values1 = np.array([[8, 16], [32, 64]], dtype=int)
    values2 = np.array([[1, 2], [1, 2]], dtype=int)
    labels = [pl.DataFrame({"x": [0, 1]}), pl.DataFrame({"y": [0, 1]})]

    array1 = create_array(values1, labels, implementation)
    array2 = create_array(values2, labels, implementation)

    result = array1 >> array2  # type: ignore

    assert_array_shape_equal(result, (2, 2))
    assert_array_values_equal(result, values1 >> values2)


# ============================================================================
# Tests for comparison operations
# ============================================================================


def test_less_than(implementation: FrameLabeledArrayImplementation) -> None:
    """Test less than comparison."""
    values1 = np.array([[1.0, 2.0], [3.0, 4.0]])
    values2 = np.array([[2.0, 2.0], [2.0, 5.0]])
    labels = [pl.DataFrame({"x": [0, 1]}), pl.DataFrame({"y": [0, 1]})]

    array1 = create_array(values1, labels, implementation)
    array2 = create_array(values2, labels, implementation)

    result = array1 < array2  # type: ignore

    assert_array_shape_equal(result, (2, 2))
    assert_array_values_equal(result, values1 < values2)


def test_less_equal(implementation: FrameLabeledArrayImplementation) -> None:
    """Test less than or equal comparison."""
    values1 = np.array([[1.0, 2.0], [3.0, 4.0]])
    values2 = np.array([[2.0, 2.0], [2.0, 5.0]])
    labels = [pl.DataFrame({"x": [0, 1]}), pl.DataFrame({"y": [0, 1]})]

    array1 = create_array(values1, labels, implementation)
    array2 = create_array(values2, labels, implementation)

    result = array1 <= array2  # type: ignore

    assert_array_shape_equal(result, (2, 2))
    assert_array_values_equal(result, values1 <= values2)


def test_greater_than(implementation: FrameLabeledArrayImplementation) -> None:
    """Test greater than comparison."""
    values1 = np.array([[1.0, 2.0], [3.0, 4.0]])
    values2 = np.array([[2.0, 2.0], [2.0, 5.0]])
    labels = [pl.DataFrame({"x": [0, 1]}), pl.DataFrame({"y": [0, 1]})]

    array1 = create_array(values1, labels, implementation)
    array2 = create_array(values2, labels, implementation)

    result = array1 > array2  # type: ignore

    assert_array_shape_equal(result, (2, 2))
    assert_array_values_equal(result, values1 > values2)


def test_greater_equal(implementation: FrameLabeledArrayImplementation) -> None:
    """Test greater than or equal comparison."""
    values1 = np.array([[1.0, 2.0], [3.0, 4.0]])
    values2 = np.array([[2.0, 2.0], [2.0, 5.0]])
    labels = [pl.DataFrame({"x": [0, 1]}), pl.DataFrame({"y": [0, 1]})]

    array1 = create_array(values1, labels, implementation)
    array2 = create_array(values2, labels, implementation)

    result = array1 >= array2  # type: ignore

    assert_array_shape_equal(result, (2, 2))
    assert_array_values_equal(result, values1 >= values2)


def test_equal_comparison(implementation: FrameLabeledArrayImplementation) -> None:
    """Test equality comparison."""
    values1 = np.array([[1.0, 2.0], [3.0, 4.0]])
    values2 = np.array([[1.0, 2.0], [3.0, 5.0]])
    labels = [pl.DataFrame({"x": [0, 1]}), pl.DataFrame({"y": [0, 1]})]

    array1 = create_array(values1, labels, implementation)
    array2 = create_array(values2, labels, implementation)

    result = array1 == array2  # type: ignore

    assert_array_shape_equal(result, (2, 2))
    assert_array_values_equal(result, values1 == values2)


def test_not_equal_comparison(implementation: FrameLabeledArrayImplementation) -> None:
    """Test not equal comparison."""
    values1 = np.array([[1.0, 2.0], [3.0, 4.0]])
    values2 = np.array([[1.0, 2.0], [3.0, 5.0]])
    labels = [pl.DataFrame({"x": [0, 1]}), pl.DataFrame({"y": [0, 1]})]

    array1 = create_array(values1, labels, implementation)
    array2 = create_array(values2, labels, implementation)

    result = array1 != array2  # type: ignore

    assert_array_shape_equal(result, (2, 2))
    assert_array_values_equal(result, values1 != values2)


# ============================================================================
# Tests for broadcasting and alignment
# ============================================================================


def test_alignment_before_operation(
    implementation: FrameLabeledArrayImplementation,
) -> None:
    """Test that alignment is performed before operations."""
    # Array with one label order
    values1 = np.array([[1.0, 2.0], [3.0, 4.0]])
    labels1 = [pl.DataFrame({"x": ["a", "b"]}), pl.DataFrame({"y": [10, 20]})]

    # Same array but with reversed first axis
    values2 = np.array([[4.0, 3.0], [2.0, 1.0]])
    labels2 = [pl.DataFrame({"x": ["b", "a"]}), pl.DataFrame({"y": [10, 20]})]

    array1 = create_array(values1, labels1, implementation)
    array2 = create_array(values2, labels2, implementation)

    result = array1 + array2  # type: ignore

    # Result should be aligned to array1's order
    assert_array_shape_equal(result, (2, 2))
    expected = np.array([[3.0, 3.0], [7.0, 7.0]])
    assert_array_values_equal(result, expected)


def test_broadcasting_scalar_preserves_array_labels(
    implementation: FrameLabeledArrayImplementation,
) -> None:
    """Test that broadcasting with scalar preserves array labels."""
    values = np.array([[1.0, 2.0], [3.0, 4.0]])
    labels = [pl.DataFrame({"x": ["a", "b"]}), pl.DataFrame({"y": [10, 20]})]

    array = create_array(values, labels, implementation)
    result = array + 5.0

    assert_array_labels_equal(result, array.labels())
    assert_array_shape_equal(result, (2, 2))
    assert_array_values_equal(result, values + 5.0)


def test_chained_arithmetic(implementation: FrameLabeledArrayImplementation) -> None:
    """Test chained arithmetic operations."""
    values1 = np.array([[1.0, 2.0], [3.0, 4.0]])
    values2 = np.array([[2.0, 1.0], [1.0, 2.0]])
    labels = [pl.DataFrame({"x": [0, 1]}), pl.DataFrame({"y": [0, 1]})]

    array1 = create_array(values1, labels, implementation)
    array2 = create_array(values2, labels, implementation)

    result = array1 + array2 * 2.0  # type: ignore

    assert_array_shape_equal(result, (2, 2))
    assert_array_values_equal(result, values1 + values2 * 2.0)


def test_mixed_array_and_scalar_operations(
    implementation: FrameLabeledArrayImplementation,
) -> None:
    """Test mixed array and scalar operations."""
    values1 = np.array([[1.0, 2.0], [3.0, 4.0]])
    values2 = np.array([[2.0, 1.0], [1.0, 2.0]])
    labels = [pl.DataFrame({"x": [0, 1]}), pl.DataFrame({"y": [0, 1]})]

    array1 = create_array(values1, labels, implementation)
    array2 = create_array(values2, labels, implementation)

    result = (array1 + 5.0) * (array2 - 1.0)  # type: ignore

    assert_array_shape_equal(result, (2, 2))
    assert_array_values_equal(result, (values1 + 5.0) * (values2 - 1.0))

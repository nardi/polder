import numpy as np
import polars as pl

import polder as pld
from polder.eager.labels import Labels
from polder.operations.align import align

from ..utils import generate_random_array, shuffle_labels

# ============================================================================
# Tests for equality comparison
# ============================================================================


def test_equals_identical_arrays(value_array_type):
    """Test that identical arrays are equal."""
    array1 = generate_random_array(value_array_type)
    array2 = generate_random_array(value_array_type)
    assert array1.equals(array2)


def test_not_equals_different_values(value_array_type):
    """Test that arrays with different values are not equal."""
    array = generate_random_array(value_array_type)
    array2 = array + 1.0
    assert not array.equals(array2)


def test_equals_after_shuffle(value_array_type):
    """Test that shuffled arrays are equal after alignment."""
    array = generate_random_array(value_array_type)
    shuffled = shuffle_labels(array, axes=[0])
    # Before alignment they're different
    assert not array.equals(shuffled)
    # After alignment they're the same
    aligned1, aligned2 = align(array, shuffled)
    assert aligned1.equals(aligned2)


# ============================================================================
# Tests for arithmetic operations
# ============================================================================


def test_add_arrays(value_array_type):
    """Test adding two arrays."""
    array1 = generate_random_array(value_array_type)
    array2 = generate_random_array(value_array_type, seed=456)
    result = array1 + array2
    assert result.shape() == array1.shape()
    assert isinstance(result.values(), value_array_type.value)
    np.testing.assert_array_almost_equal(
        result.values(), array1.values() + array2.values()
    )


def test_add_scalar_to_array(value_array_type):
    """Test adding a scalar to an array."""
    array = generate_random_array(value_array_type)
    scalar = 5.0
    result = array + scalar
    assert result.shape() == array.shape()
    assert isinstance(result.values(), value_array_type.value)
    np.testing.assert_array_almost_equal(result.values(), array.values() + scalar)


def test_add_array_to_scalar(value_array_type):
    """Test adding an array to a scalar (reflected operation)."""
    array = generate_random_array(value_array_type)
    scalar = 3.0
    result = scalar + array
    assert result.shape() == array.shape()
    assert isinstance(result.values(), value_array_type.value)
    np.testing.assert_array_almost_equal(result.values(), scalar + array.values())


def test_subtract_arrays(value_array_type):
    """Test subtracting arrays."""
    array1 = generate_random_array(value_array_type)
    array2 = generate_random_array(value_array_type, seed=456)
    result = array1 - array2
    assert result.shape() == array1.shape()
    assert isinstance(result.values(), value_array_type.value)
    np.testing.assert_array_almost_equal(
        result.values(), array1.values() - array2.values()
    )


def test_subtract_scalar_from_array(value_array_type):
    """Test subtracting a scalar from an array."""
    array = generate_random_array(value_array_type)
    scalar = 2.0
    result = array - scalar
    assert isinstance(result.values(), value_array_type.value)
    np.testing.assert_array_almost_equal(result.values(), array.values() - scalar)


def test_subtract_with_reflected(value_array_type):
    """Test subtraction with reflected operation (scalar - array)."""
    array = generate_random_array(value_array_type)
    scalar = 2.0
    result = scalar - array
    # Verify the operation works (exact semantics depend on alignment)
    assert result.shape() == array.shape()
    assert isinstance(result.values(), value_array_type.value)
    assert isinstance(result, type(array))


def test_multiply_arrays(value_array_type):
    """Test multiplying arrays."""
    array1 = generate_random_array(value_array_type)
    array2 = generate_random_array(value_array_type, seed=456)
    result = array1 * array2
    assert result.shape() == array1.shape()
    assert isinstance(result.values(), value_array_type.value)
    np.testing.assert_array_almost_equal(
        result.values(), array1.values() * array2.values()
    )


def test_multiply_scalar_to_array(value_array_type):
    """Test multiplying an array by a scalar."""
    array = generate_random_array(value_array_type)
    scalar = 4.0
    result = array * scalar
    assert isinstance(result.values(), value_array_type.value)
    np.testing.assert_array_almost_equal(result.values(), array.values() * scalar)


def test_multiply_array_by_scalar(value_array_type):
    """Test multiplying a scalar by an array (reflected operation)."""
    array = generate_random_array(value_array_type)
    scalar = 4.0
    result = scalar * array
    assert isinstance(result.values(), value_array_type.value)
    np.testing.assert_array_almost_equal(result.values(), scalar * array.values())


def test_divide_arrays(value_array_type):
    """Test dividing arrays."""
    array1 = generate_random_array(value_array_type)
    array2 = (
        generate_random_array(value_array_type, seed=456) + 1.0
    )  # Avoid division by zero
    result = array1 / array2
    assert result.shape() == array1.shape()
    assert isinstance(result.values(), value_array_type.value)
    np.testing.assert_array_almost_equal(
        result.values(), array1.values() / array2.values()
    )


def test_divide_array_by_scalar(value_array_type):
    """Test dividing an array by a scalar."""
    array = generate_random_array(value_array_type)
    scalar = 2.0
    result = array / scalar
    assert isinstance(result.values(), value_array_type.value)
    np.testing.assert_array_almost_equal(result.values(), array.values() / scalar)


def test_divide_with_reflected(value_array_type):
    """Test division with reflected operation (scalar / array)."""
    array = generate_random_array(value_array_type) + 1.0  # Avoid division by zero
    scalar = 2.0
    result = scalar / array
    assert result.shape() == array.shape()
    assert isinstance(result.values(), value_array_type.value)
    np.testing.assert_array_almost_equal(result.values(), scalar / array.values())


def test_floor_divide_arrays(value_array_type):
    """Test floor dividing arrays."""
    array1 = generate_random_array(value_array_type) * 10  # Scale to get integers
    array2 = (
        generate_random_array(value_array_type, seed=456) * 10 + 1.0
    )  # Avoid division by zero
    result = array1 // array2
    assert result.shape() == array1.shape()
    assert isinstance(result.values(), value_array_type.value)
    np.testing.assert_array_almost_equal(
        result.values(), np.floor(array1.values() / array2.values())
    )


def test_modulo_arrays(value_array_type):
    """Test modulo operation on arrays."""
    array1 = generate_random_array(value_array_type) * 10  # Scale to get integers
    array2 = generate_random_array(value_array_type, seed=456) * 10 + 1.0
    result = array1 % array2
    assert result.shape() == array1.shape()
    assert isinstance(result.values(), value_array_type.value)


def test_power_arrays():
    """Test power operation on arrays."""
    # TODO: parametrize over value array types.
    np.random.seed(123)
    values1 = np.abs(np.random.randn(2, 3)) + 0.5  # Ensure positive values
    values2 = np.abs(np.random.randn(2, 3))

    labels = [pl.DataFrame({"i": np.arange(2)}), pl.DataFrame({"j": np.arange(3)})]
    array1 = pld.from_values_and_labels(values1, labels)
    array2 = pld.from_values_and_labels(
        values2, [pl.DataFrame({"i": np.arange(2)}), pl.DataFrame({"j": np.arange(3)})]
    )

    result = array1**array2
    assert result.shape() == array1.shape()


# ============================================================================
# Tests for bitwise operations
# ============================================================================


def test_bitwise_and():
    """Test bitwise AND operation."""
    # TODO: parametrize over value array types.
    np.random.seed(123)
    values1 = np.random.randint(0, 256, size=(2, 3))
    values2 = np.random.randint(0, 256, size=(2, 3))

    labels = [pl.DataFrame({"i": np.arange(2)}), pl.DataFrame({"j": np.arange(3)})]
    array1 = pld.from_values_and_labels(values1, labels)
    array2 = pld.from_values_and_labels(
        values2, [pl.DataFrame({"i": np.arange(2)}), pl.DataFrame({"j": np.arange(3)})]
    )

    result = array1 & array2
    np.testing.assert_array_equal(result.values(), values1 & values2)


def test_bitwise_or():
    """Test bitwise OR operation."""
    # TODO: parametrize over value array types.
    np.random.seed(123)
    values1 = np.random.randint(0, 256, size=(2, 3))
    values2 = np.random.randint(0, 256, size=(2, 3))

    labels = [pl.DataFrame({"i": np.arange(2)}), pl.DataFrame({"j": np.arange(3)})]
    array1 = pld.from_values_and_labels(values1, labels)
    array2 = pld.from_values_and_labels(
        values2, [pl.DataFrame({"i": np.arange(2)}), pl.DataFrame({"j": np.arange(3)})]
    )

    result = array1 | array2
    np.testing.assert_array_equal(result.values(), values1 | values2)


def test_bitwise_xor():
    """Test bitwise XOR operation."""
    # TODO: parametrize over value array types.
    np.random.seed(123)
    values1 = np.random.randint(0, 256, size=(2, 3))
    values2 = np.random.randint(0, 256, size=(2, 3))

    labels = [pl.DataFrame({"i": np.arange(2)}), pl.DataFrame({"j": np.arange(3)})]
    array1 = pld.from_values_and_labels(values1, labels)
    array2 = pld.from_values_and_labels(
        values2, [pl.DataFrame({"i": np.arange(2)}), pl.DataFrame({"j": np.arange(3)})]
    )

    result = array1 ^ array2
    np.testing.assert_array_equal(result.values(), values1 ^ values2)


# ============================================================================
# Tests for comparison operations
# ============================================================================


def test_less_than(value_array_type):
    """Test less than comparison."""
    array1 = generate_random_array(value_array_type)
    array2 = generate_random_array(value_array_type, seed=456)
    result = array1 < array2
    assert result.shape() == array1.shape()
    assert isinstance(result.values(), value_array_type.value)
    np.testing.assert_array_equal(result.values(), array1.values() < array2.values())


def test_less_equal(value_array_type):
    """Test less than or equal comparison."""
    array1 = generate_random_array(value_array_type)
    array2 = generate_random_array(value_array_type, seed=456)
    result = array1 <= array2
    assert isinstance(result.values(), value_array_type.value)
    np.testing.assert_array_equal(result.values(), array1.values() <= array2.values())


def test_greater_than(value_array_type):
    """Test greater than comparison."""
    array1 = generate_random_array(value_array_type)
    array2 = generate_random_array(value_array_type, seed=456)
    result = array1 > array2
    assert isinstance(result.values(), value_array_type.value)
    np.testing.assert_array_equal(result.values(), array1.values() > array2.values())


def test_greater_equal(value_array_type):
    """Test greater than or equal comparison."""
    array1 = generate_random_array(value_array_type)
    array2 = generate_random_array(value_array_type, seed=456)
    result = array1 >= array2
    assert isinstance(result.values(), value_array_type.value)
    np.testing.assert_array_equal(result.values(), array1.values() >= array2.values())


def test_equal_comparison(value_array_type):
    """Test equality comparison (element-wise)."""
    array1 = generate_random_array(value_array_type)
    array2 = generate_random_array(value_array_type, seed=123)  # Same seed
    result = array1 == array2
    assert isinstance(result.values(), value_array_type.value)
    np.testing.assert_array_equal(result.values(), array1.values() == array2.values())


def test_not_equal_comparison(value_array_type):
    """Test not equal comparison (element-wise)."""
    array1 = generate_random_array(value_array_type)
    array2 = generate_random_array(value_array_type, seed=456)
    result = array1 != array2
    assert isinstance(result.values(), value_array_type.value)
    np.testing.assert_array_equal(result.values(), array1.values() != array2.values())


# ============================================================================
# Tests for broadcasting and alignment
# ============================================================================


def test_alignment_before_operation(value_array_type):
    """Test that alignment works correctly before binary operations."""
    array = generate_random_array(value_array_type)
    shuffled = shuffle_labels(array, axes=[0])

    # Operations should work because alignment is automatic
    result = array + shuffled
    assert result.shape() == array.shape()
    assert isinstance(result.values(), value_array_type.value)


def test_broadcasting_scalar_preserves_array_labels(value_array_type):
    """Test that broadcasting a scalar preserves the array's labels."""
    array = generate_random_array(value_array_type)
    result = array + 5.0
    assert isinstance(result.values(), value_array_type.value)

    # Check that the labels are preserved
    original_labels = array.labels()
    result_labels = result.labels()

    # The labels should still be present
    assert len(result_labels) == len(original_labels)
    assert all(l1 is l2 for l1, l2 in zip(result_labels, original_labels))


def test_operation_with_shuffled_axes(value_array_type):
    """Test binary operations with shuffled axes to verify alignment."""
    np.random.seed(123)
    array1 = generate_random_array(value_array_type, shape=(3, 4))

    # Shuffle the first axis
    shuffled = shuffle_labels(array1, axes=[0])

    # After aligning, the sum should work correctly
    result = array1 + shuffled
    assert result.shape() == array1.shape()
    assert isinstance(result.values(), value_array_type.value)

    # The result should align with the original array
    aligned1, aligned2 = align(array1, shuffled)
    assert isinstance(aligned1.values(), value_array_type.value)
    assert isinstance(aligned2.values(), value_array_type.value)
    expected = aligned1.values() + aligned2.values()
    np.testing.assert_array_almost_equal(result.values(), expected)


# ============================================================================
# Tests for complex operations
# ============================================================================


def test_chained_arithmetic(value_array_type):
    """Test chaining multiple arithmetic operations."""
    array = generate_random_array(value_array_type)
    result = (array + 2.0) * 3.0 - 1.0
    expected = (array.values() + 2.0) * 3.0 - 1.0
    np.testing.assert_array_almost_equal(result.values(), expected)


def test_mixed_array_and_scalar_operations(value_array_type):
    """Test mixing array-array and array-scalar operations."""
    array1 = generate_random_array(value_array_type)
    array2 = generate_random_array(value_array_type, seed=456)

    result = (array1 + 5.0) * array2 - 2.0
    assert isinstance(result.values(), value_array_type.value)
    expected = (array1.values() + 5.0) * array2.values() - 2.0
    np.testing.assert_array_almost_equal(result.values(), expected)


def test_matmul_2d_arrays():
    """Test matrix multiplication of two 2D arrays."""
    # TODO: parametrize over value array types.
    values1 = np.random.randn(2, 3)
    values2 = np.random.randn(3, 4)

    labels1 = [pl.DataFrame({"i": np.arange(2)}), pl.DataFrame({"j": np.arange(3)})]
    labels2 = [pl.DataFrame({"j": np.arange(3)}), pl.DataFrame({"k": np.arange(4)})]

    array1 = pld.from_values_and_labels(values1, labels1)
    array2 = pld.from_values_and_labels(values2, labels2)

    result = array1 @ array2
    expected = values1 @ values2

    assert result.shape() == (2, 4)
    np.testing.assert_array_almost_equal(result.values(), expected)

    # Make sure that shuffling the labels makes no difference.
    result2 = shuffle_labels(array1) @ shuffle_labels(array2)
    assert result2.shape() == (2, 4)
    _, result2 = align(result, result2)
    np.testing.assert_array_almost_equal(result2.values(), result.values())


def test_broadcasting_with_unlabeled_axes(value_array_type):
    # Create two 1-dimensional arrays (vectors).
    array1 = generate_random_array(value_array_type, shape=(4,), seed=123)
    array2 = generate_random_array(value_array_type, shape=(3,), seed=456)

    # Calculate the outer product of the two vectors using broadcasting.
    result = array1[:, None] * array2[None, :]

    assert result.shape() == (4, 3)
    assert Labels(result.labels()) == Labels([array1.labels(0), array2.labels(0)])
    assert isinstance(result.values(), value_array_type.value)
    np.testing.assert_array_equal(
        result.values(), array1.values[:, None] * array2.values[None, :]
    )

    # Make sure we can also broadcast an unlabeled axis with a labeled one.
    result2 = result + array1[None, :-1]

    assert result2.shape() == (4, 3)
    assert Labels(result2.labels()) == Labels(result.labels())
    assert isinstance(result2.values(), value_array_type.value)
    np.testing.assert_array_equal(
        result2.values(), result.values() + array1.values[None, :-1]
    )

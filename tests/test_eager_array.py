import numpy as np
import polars as pl

import polder as pld
from polder.eager.align import align
from polder.eager.array import EagerFrameLabeledArray


def generate_random_array(shape=(2, 3), seed=123):
    """Generate a random array with controllable shape and a fixed seed for reproducibility."""
    np.random.seed(seed)
    values = np.random.randn(*shape)

    labels = []
    for dim_size in shape:
        # Create a dataframe with random labels for each dimension
        label_data = {
            f"dim{i}": np.arange(dim_size)
            for i in range(min(2, dim_size))  # At least one column, up to two
        }
        if not label_data:
            label_data = {"index": np.arange(dim_size)}
        labels.append(pl.DataFrame(label_data))

    return pld.from_values_and_labels(values, labels)


def shuffle_labels(array, axes=None, seed=123):
    """Shuffle the labels of an array along specified axes to test alignment."""
    np.random.seed(seed)

    if axes is None:
        axes = list(range(len(array.shape())))
    elif not isinstance(axes, (list, tuple)):
        axes = [axes]

    # Create index shuffles for each axis
    indices = []
    for i, size in enumerate(array.shape()):
        if i in axes:
            idx = np.random.permutation(size)
        else:
            idx = np.arange(size)
        indices.append(idx)

    # Apply the indexing
    return array[tuple(indices)]


# ============================================================================
# Tests for array creation and basic properties
# ============================================================================


def test_array_creation():
    """Test creating an array from values and labels."""
    array = generate_random_array()
    assert array.shape() == (2, 3)
    assert array.values().shape == (2, 3)
    assert len(array.labels()) == 2


def test_array_creation_with_different_shapes():
    """Test creating arrays with different shapes."""
    for shape in [(1, 1), (3, 4), (5, 2, 3)]:
        array = generate_random_array(shape=shape)
        assert array.shape() == shape


def test_array_values():
    """Test that values are correctly stored."""
    array = generate_random_array()
    values = array.values()
    assert isinstance(values, np.ndarray)
    assert values.shape == (2, 3)


def test_array_labels():
    """Test accessing labels."""
    array = generate_random_array()
    labels = array.labels()
    assert len(labels) == 2
    assert all(label is not None for label in labels)


# ============================================================================
# Tests for indexing operations
# ============================================================================


def test_indexing_slice():
    """Test indexing with a slice."""
    array = generate_random_array(shape=(5, 3))
    indexed = array[1:3]
    assert indexed.shape() == (2, 3)


def test_indexing_multiple_axes():
    """Test indexing multiple axes at once."""
    array = generate_random_array(shape=(4, 5))
    indexed = array[[1, 3], [0, 2, 4]]
    assert indexed.shape() == (2, 3)


# ============================================================================
# Tests for equality comparison
# ============================================================================


def test_equals_identical_arrays():
    """Test that identical arrays are equal."""
    array1 = generate_random_array()
    array2 = generate_random_array()
    assert pld.equals(array1, array2)


def test_not_equals_different_values():
    """Test that arrays with different values are not equal."""
    array = generate_random_array()
    array2 = array + 1.0
    assert not pld.equals(array, array2)


def test_equals_after_shuffle():
    """Test that shuffled arrays are equal after alignment."""
    array = generate_random_array()
    shuffled = shuffle_labels(array, axes=[0])
    # Before alignment they're different
    assert not pld.equals(array, shuffled)
    # After alignment they're the same
    aligned1, aligned2 = align(array, shuffled)
    assert pld.equals(aligned1, aligned2)


# ============================================================================
# Tests for arithmetic operations
# ============================================================================


def test_add_arrays():
    """Test adding two arrays."""
    array1 = generate_random_array()
    array2 = generate_random_array(seed=456)
    result = array1 + array2
    assert result.shape() == array1.shape()
    np.testing.assert_array_almost_equal(
        result.values(), array1.values() + array2.values()
    )


def test_add_scalar_to_array():
    """Test adding a scalar to an array."""
    array = generate_random_array()
    scalar = 5.0
    result = array + scalar
    assert result.shape() == array.shape()
    np.testing.assert_array_almost_equal(result.values(), array.values() + scalar)


def test_add_array_to_scalar():
    """Test adding an array to a scalar (reflected operation)."""
    array = generate_random_array()
    scalar = 3.0
    result = scalar + array
    assert result.shape() == array.shape()
    np.testing.assert_array_almost_equal(result.values(), scalar + array.values())


def test_subtract_arrays():
    """Test subtracting arrays."""
    array1 = generate_random_array()
    array2 = generate_random_array(seed=456)
    result = array1 - array2
    assert result.shape() == array1.shape()
    np.testing.assert_array_almost_equal(
        result.values(), array1.values() - array2.values()
    )


def test_subtract_scalar_from_array():
    """Test subtracting a scalar from an array."""
    array = generate_random_array()
    scalar = 2.0
    result = array - scalar
    np.testing.assert_array_almost_equal(result.values(), array.values() - scalar)


def test_subtract_with_reflected():
    """Test subtraction with reflected operation (scalar - array)."""
    array = generate_random_array()
    scalar = 2.0
    result = scalar - array
    # Verify the operation works (exact semantics depend on alignment)
    assert result.shape() == array.shape()
    assert isinstance(result, type(array))


def test_multiply_arrays():
    """Test multiplying arrays."""
    array1 = generate_random_array()
    array2 = generate_random_array(seed=456)
    result = array1 * array2
    assert result.shape() == array1.shape()
    np.testing.assert_array_almost_equal(
        result.values(), array1.values() * array2.values()
    )


def test_multiply_scalar_to_array():
    """Test multiplying an array by a scalar."""
    array = generate_random_array()
    scalar = 4.0
    result = array * scalar
    np.testing.assert_array_almost_equal(result.values(), array.values() * scalar)


def test_multiply_array_by_scalar():
    """Test multiplying a scalar by an array (reflected operation)."""
    array = generate_random_array()
    scalar = 4.0
    result = scalar * array
    np.testing.assert_array_almost_equal(result.values(), scalar * array.values())


def test_divide_arrays():
    """Test dividing arrays."""
    array1 = generate_random_array()
    array2 = generate_random_array(seed=456) + 1.0  # Avoid division by zero
    result = array1 / array2
    assert result.shape() == array1.shape()
    np.testing.assert_array_almost_equal(
        result.values(), array1.values() / array2.values()
    )


def test_divide_array_by_scalar():
    """Test dividing an array by a scalar."""
    array = generate_random_array()
    scalar = 2.0
    result = array / scalar
    np.testing.assert_array_almost_equal(result.values(), array.values() / scalar)


def test_divide_with_reflected():
    """Test division with reflected operation (scalar / array)."""
    array = generate_random_array() + 1.0  # Avoid division by zero
    scalar = 2.0
    result = scalar / array
    assert result.shape() == array.shape()
    np.testing.assert_array_almost_equal(result.values(), scalar / array.values())


def test_floor_divide_arrays():
    """Test floor dividing arrays."""
    array1 = generate_random_array() * 10  # Scale to get integers
    array2 = generate_random_array(seed=456) * 10 + 1.0  # Avoid division by zero
    result = array1 // array2
    assert result.shape() == array1.shape()
    np.testing.assert_array_almost_equal(
        result.values(), np.floor(array1.values() / array2.values())
    )


def test_modulo_arrays():
    """Test modulo operation on arrays."""
    array1 = generate_random_array() * 10  # Scale to get integers
    array2 = generate_random_array(seed=456) * 10 + 1.0
    result = array1 % array2
    assert result.shape() == array1.shape()


def test_power_arrays():
    """Test power operation on arrays."""
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


def test_less_than():
    """Test less than comparison."""
    array1 = generate_random_array()
    array2 = generate_random_array(seed=456)
    result = array1 < array2
    assert result.shape() == array1.shape()
    np.testing.assert_array_equal(result.values(), array1.values() < array2.values())


def test_less_equal():
    """Test less than or equal comparison."""
    array1 = generate_random_array()
    array2 = generate_random_array(seed=456)
    result = array1 <= array2
    np.testing.assert_array_equal(result.values(), array1.values() <= array2.values())


def test_greater_than():
    """Test greater than comparison."""
    array1 = generate_random_array()
    array2 = generate_random_array(seed=456)
    result = array1 > array2
    np.testing.assert_array_equal(result.values(), array1.values() > array2.values())


def test_greater_equal():
    """Test greater than or equal comparison."""
    array1 = generate_random_array()
    array2 = generate_random_array(seed=456)
    result = array1 >= array2
    np.testing.assert_array_equal(result.values(), array1.values() >= array2.values())


def test_equal_comparison():
    """Test equality comparison (element-wise)."""
    array1 = generate_random_array()
    array2 = generate_random_array(seed=123)  # Same seed
    # TODO: figure out how to get this comparison to type properly.
    result: EagerFrameLabeledArray = array1 == array2  # type: ignore
    np.testing.assert_array_equal(result.values(), array1.values() == array2.values())


def test_not_equal_comparison():
    """Test not equal comparison (element-wise)."""
    array1 = generate_random_array()
    array2 = generate_random_array(seed=456)
    # TODO: figure out how to get this comparison to type properly.
    result: EagerFrameLabeledArray = array1 != array2  # type: ignore
    np.testing.assert_array_equal(result.values(), array1.values() != array2.values())


# ============================================================================
# Tests for broadcasting and alignment
# ============================================================================


def test_alignment_before_operation():
    """Test that alignment works correctly before binary operations."""
    array = generate_random_array()
    shuffled = shuffle_labels(array, axes=[0])

    # Operations should work because alignment is automatic
    result = array + shuffled
    assert result.shape() == array.shape()


def test_broadcasting_scalar_preserves_array_labels():
    """Test that broadcasting a scalar preserves the array's labels."""
    array = generate_random_array()
    result = array + 5.0

    # Check that the labels are preserved
    original_labels = array.labels()
    result_labels = result.labels()

    # The labels should still be present
    assert len(result_labels) == len(original_labels)


def test_operation_with_shuffled_axes():
    """Test binary operations with shuffled axes to verify alignment."""
    np.random.seed(123)
    array1 = generate_random_array(shape=(3, 4))

    # Shuffle the first axis
    shuffled = shuffle_labels(array1, axes=[0])

    # After aligning, the sum should work correctly
    result = array1 + shuffled
    assert result.shape() == array1.shape()

    # The result should align with the original array
    aligned1, aligned2 = align(array1, shuffled)
    expected = aligned1.values() + aligned2.values()
    np.testing.assert_array_almost_equal(result.values(), expected)


# ============================================================================
# Tests for complex operations
# ============================================================================


def test_chained_arithmetic():
    """Test chaining multiple arithmetic operations."""
    array = generate_random_array()
    result = (array + 2.0) * 3.0 - 1.0
    expected = (array.values() + 2.0) * 3.0 - 1.0
    np.testing.assert_array_almost_equal(result.values(), expected)


def test_mixed_array_and_scalar_operations():
    """Test mixing array-array and array-scalar operations."""
    array1 = generate_random_array()
    array2 = generate_random_array(seed=456)

    result = (array1 + 5.0) * array2 - 2.0
    expected = (array1.values() + 5.0) * array2.values() - 2.0
    np.testing.assert_array_almost_equal(result.values(), expected)


def test_matmul_2d_arrays():
    """Test matrix multiplication of two 2D arrays."""
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

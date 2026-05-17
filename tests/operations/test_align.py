"""Tests for align operation dispatched from polder.operations.align."""

from collections.abc import Sequence
from enum import Enum

import narwhals.typing as nwt
import numpy as np
import polars as pl
import pytest

import polder as pld
from polder.eager.labels import Labels
from polder.operations.align import align
from polder.protocols.array import FrameLabeledArray
from polder.protocols.implementations import EAGER, LAZY


class ImplementationType(Enum):
    EAGER = EAGER
    LAZY = LAZY


@pytest.fixture(params=[ImplementationType.EAGER, ImplementationType.LAZY])
def implementation(request):
    """Parametrized fixture for testing both eager and lazy implementations."""
    return request.param


def create_array(
    values: np.ndarray,
    labels: Sequence[pl.DataFrame | None],
    implementation: ImplementationType,
):
    """Create either an eager or lazy array depending on implementation."""
    return pld.from_values_and_labels(
        values, labels, implementation=implementation.value
    )


def assert_array_labels_equal(
    result: FrameLabeledArray, expected: Sequence[nwt.DataFrameT | None]
) -> None:
    """Assert that array labels match expected (handling lazy evaluation)."""
    assert Labels(result.labels()) == Labels(expected)


def assert_array_values_equal(result: FrameLabeledArray, expected: np.ndarray) -> None:
    """Assert that array values match expected (handling lazy evaluation)."""
    np.testing.assert_array_equal(result.values(), expected)


def assert_array_shape_equal(
    result: FrameLabeledArray, expected: tuple[int, ...]
) -> None:
    """Assert that array shape matches expected."""
    assert result.shape() == expected


# ============================================================================
# Tests for basic alignment
# ============================================================================


def test_align_identical_labels(implementation: ImplementationType) -> None:
    """Test aligning arrays with identical labels."""
    values1 = np.array([[1.0, 2.0], [3.0, 4.0]])
    values2 = np.array([[5.0, 6.0], [7.0, 8.0]])
    labels = [pl.DataFrame({"x": [0, 1]}), pl.DataFrame({"y": [0, 1]})]

    array1 = create_array(values1, labels, implementation)
    array2 = create_array(values2, labels, implementation)

    result1, result2 = align(array1, array2)

    # Both should maintain their values since labels are identical
    assert_array_labels_equal(result1, result2.labels())
    assert_array_shape_equal(result1, (2, 2))
    assert_array_shape_equal(result2, (2, 2))
    assert_array_values_equal(result1, values1)
    assert_array_values_equal(result2, values2)


def test_align_reordered_labels(implementation: ImplementationType) -> None:
    """Test aligning arrays with same labels but different order."""
    values1 = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    labels1 = [pl.DataFrame({"x": ["a", "b", "c"]}), pl.DataFrame({"y": [10, 20]})]

    # Reorder the first array's labels
    values2 = np.array([[5.0, 6.0], [1.0, 2.0], [3.0, 4.0]])
    labels2 = [pl.DataFrame({"x": ["c", "a", "b"]}), pl.DataFrame({"y": [10, 20]})]

    array1 = create_array(values1, labels1, implementation)
    array2 = create_array(values2, labels2, implementation)

    result1, result2 = align(array1, array2)

    # Both should be aligned to the first array's order
    assert_array_labels_equal(result1, result2.labels())
    assert_array_shape_equal(result1, (3, 2))
    assert_array_shape_equal(result2, (3, 2))
    assert_array_values_equal(result1, values1)
    # result2 should be reordered to match values1's label order
    assert_array_values_equal(result2, np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))


def test_align_single_array(implementation: ImplementationType) -> None:
    """Test aligning a single array (identity operation)."""
    values = np.array([[1.0, 2.0], [3.0, 4.0]])
    labels = [pl.DataFrame({"x": [0, 1]}), pl.DataFrame({"y": [0, 1]})]

    array = create_array(values, labels, implementation)
    (result,) = align(array)

    assert_array_labels_equal(result, array.labels())
    assert_array_shape_equal(result, (2, 2))
    assert_array_values_equal(result, values)


def test_align_empty_tuple(implementation: ImplementationType) -> None:
    """Test aligning no arrays returns empty tuple."""
    result = align()
    assert result == ()


# ============================================================================
# Tests for multi-dimensional alignment
# ============================================================================


def test_align_3d_arrays(implementation: ImplementationType) -> None:
    """Test aligning 3D arrays."""
    values1 = np.arange(24).reshape(2, 3, 4).astype(float)
    values2 = np.arange(24, 48).reshape(2, 3, 4).astype(float)
    labels = [
        pl.DataFrame({"i": [0, 1]}),
        pl.DataFrame({"j": [0, 1, 2]}),
        pl.DataFrame({"k": [0, 1, 2, 3]}),
    ]

    array1 = create_array(values1, labels, implementation)
    array2 = create_array(values2, labels, implementation)

    result1, result2 = align(array1, array2)

    assert_array_labels_equal(result1, result2.labels())
    assert_array_shape_equal(result1, (2, 3, 4))
    assert_array_shape_equal(result2, (2, 3, 4))
    assert_array_values_equal(result1, values1)
    assert_array_values_equal(result2, values2)


def test_align_mixed_label_types(implementation: ImplementationType) -> None:
    """Test aligning arrays with different label column types."""
    values1 = np.array([[1.0, 2.0], [3.0, 4.0]])
    labels1 = [pl.DataFrame({"x": ["a", "b"]}), pl.DataFrame({"y": [10, 20]})]

    values2 = np.array([[5.0, 6.0], [7.0, 8.0]])
    labels2 = [pl.DataFrame({"x": ["a", "b"]}), pl.DataFrame({"y": [10, 20]})]

    array1 = create_array(values1, labels1, implementation)
    array2 = create_array(values2, labels2, implementation)

    result1, result2 = align(array1, array2)

    assert_array_labels_equal(result1, result2.labels())
    assert_array_shape_equal(result1, (2, 2))
    assert_array_shape_equal(result2, (2, 2))


# ============================================================================
# Tests for alignment with None labels
# ============================================================================


def test_align_partial_none_labels(implementation: ImplementationType) -> None:
    """Test aligning arrays where one has labeled axis and other doesn't."""
    # Array with size-1 axis with label
    values1 = np.array([[1.0, 2.0, 3.0]])
    labels1 = [None, pl.DataFrame({"y": [0, 1, 2]})]

    # Array with regular labeled axes
    values2 = np.array([[4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    labels2 = [pl.DataFrame({"x": [0, 1]}), pl.DataFrame({"y": [2, 1, 0]})]

    array1 = create_array(values1, labels1, implementation)
    array2 = create_array(values2, labels2, implementation)

    result1, result2 = align(array1, array2)

    # The first axis should remain as-is, while the second array has the second
    # axis flipped.
    assert_array_labels_equal(result1, array1.labels())
    result2_labels = list(array2.labels())
    assert result2_labels[1] is not None
    result2_labels[1] = result2_labels[1].sort(by="y")
    assert_array_labels_equal(result2, result2_labels)
    assert_array_values_equal(result1, values1)
    assert_array_values_equal(result2, values2[:, ::-1])


# ============================================================================
# Tests for alignment failure cases
# ============================================================================


def test_align_incompatible_dimensions_raises(
    implementation: ImplementationType,
) -> None:
    """Test that aligning arrays with different dimensions raises error."""
    values1 = np.array([[1.0, 2.0], [3.0, 4.0]])
    labels1 = [pl.DataFrame({"x": [0, 1]}), pl.DataFrame({"y": [0, 1]})]

    values2 = np.array([1.0, 2.0, 3.0])
    labels2 = [pl.DataFrame({"x": [0, 1, 2]})]

    array1 = create_array(values1, labels1, implementation)
    array2 = create_array(values2, labels2, implementation)

    with pytest.raises(Exception):
        result1, result2 = align(array1, array2)
        # Lazy arrays will only error when their shape is resolved.
        result1.shape()
        result2.shape()


def test_align_incompatible_axis_lengths_raises(
    implementation: ImplementationType,
) -> None:
    """Test that aligning arrays with incompatible axis lengths raises error."""
    values1 = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    labels1 = [pl.DataFrame({"x": [0, 1, 2]}), pl.DataFrame({"y": [0, 1]})]

    values2 = np.array([[1.0, 2.0], [3.0, 4.0]])
    labels2 = [pl.DataFrame({"x": [0, 1]}), pl.DataFrame({"y": [0, 1]})]

    array1 = create_array(values1, labels1, implementation)
    array2 = create_array(values2, labels2, implementation)

    with pytest.raises(Exception):
        result1, result2 = align(array1, array2)
        # Lazy arrays will only error when their values are resolved.
        # TODO: perhaps they should also error upon calling `labels`?
        result1.values()
        result2.values()


def test_align_different_label_columns_raises(
    implementation: ImplementationType,
) -> None:
    """Test that arrays with different label columns cannot be aligned."""
    values1 = np.array([[1.0, 2.0], [3.0, 4.0]])
    labels1 = [pl.DataFrame({"x": [0, 1]}), pl.DataFrame({"y": [0, 1]})]

    values2 = np.array([[5.0, 6.0], [7.0, 8.0]])
    labels2 = [pl.DataFrame({"z": [0, 1]}), pl.DataFrame({"y": [0, 1]})]

    array1 = create_array(values1, labels1, implementation)
    array2 = create_array(values2, labels2, implementation)

    with pytest.raises(Exception):
        align(array1, array2)
        # Note: lazy arrays will error on `align`, since the label frames have
        # different schemas.


# ============================================================================
# Tests for alignment with multiple arrays
# ============================================================================


def test_align_three_arrays(implementation: ImplementationType) -> None:
    """Test aligning three arrays simultaneously."""
    values1 = np.array([[1.0, 2.0], [3.0, 4.0]])
    values2 = np.array([[5.0, 6.0], [7.0, 8.0]])
    values3 = np.array([[9.0, 10.0], [11.0, 12.0]])
    labels = [pl.DataFrame({"x": [0, 1]}), pl.DataFrame({"y": [0, 1]})]

    array1 = create_array(values1, labels, implementation)
    array2 = create_array(values2, labels, implementation)
    array3 = create_array(values3, labels, implementation)

    result1, result2, result3 = align(array1, array2, array3)

    assert_array_shape_equal(result1, (2, 2))
    assert_array_shape_equal(result2, (2, 2))
    assert_array_shape_equal(result3, (2, 2))
    assert_array_values_equal(result1, values1)
    assert_array_values_equal(result2, values2)
    assert_array_values_equal(result3, values3)


def test_align_multiple_reordered_arrays(
    implementation: ImplementationType,
) -> None:
    """Test aligning multiple arrays with different label orders."""
    values1 = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    labels1 = [pl.DataFrame({"x": ["a", "b", "c"]}), pl.DataFrame({"y": [10, 20]})]

    values2 = np.array([[3.0, 4.0], [5.0, 6.0], [1.0, 2.0]])
    labels2 = [pl.DataFrame({"x": ["b", "c", "a"]}), pl.DataFrame({"y": [10, 20]})]

    values3 = np.array([[5.0, 6.0], [1.0, 2.0], [3.0, 4.0]])
    labels3 = [pl.DataFrame({"x": ["c", "a", "b"]}), pl.DataFrame({"y": [10, 20]})]

    array1 = create_array(values1, labels1, implementation)
    array2 = create_array(values2, labels2, implementation)
    array3 = create_array(values3, labels3, implementation)

    result1, result2, result3 = align(array1, array2, array3)

    # All should be aligned to first array's order
    expected = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    assert_array_values_equal(result1, expected)
    assert_array_values_equal(result2, expected)
    assert_array_values_equal(result3, expected)


# ============================================================================
# Tests for label inspection after alignment
# ============================================================================


def test_align_preserves_label_structure(implementation: ImplementationType) -> None:
    """Test that alignment preserves the label frame structure."""
    values1 = np.array([[1.0, 2.0], [3.0, 4.0]])
    labels = [
        pl.DataFrame({"x": [0, 1], "x_name": ["first", "second"]}),
        pl.DataFrame({"y": [10, 20]}),
    ]

    array = create_array(values1, labels, implementation)
    (result,) = align(array)

    # Labels should be preserved
    result_labels = result.labels()
    assert result_labels[0] is not None
    assert result_labels[1] is not None
    assert "x" in result_labels[0].columns
    assert "x_name" in result_labels[0].columns


def test_align_axis_parameter_single_axis(implementation: ImplementationType) -> None:
    """Test align with explicit axes parameter for single axis."""
    values1 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    values2 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    labels = [pl.DataFrame({"x": [0, 1]}), pl.DataFrame({"y": [0, 1, 2]})]

    array1 = create_array(values1, labels, implementation)
    array2 = create_array(values2, labels, implementation)

    # Align only the first axis
    result1, result2 = align(array1, array2, axes=[(0, 0)])

    assert_array_shape_equal(result1, (2, 3))
    assert_array_shape_equal(result2, (2, 3))


def test_align_axis_parameter_multiple_axes(implementation: ImplementationType) -> None:
    """Test align with explicit axes parameter for multiple axes."""
    values1 = np.arange(24).reshape(2, 3, 4).astype(float)
    values2 = np.arange(24, 48).reshape(2, 3, 4).astype(float)
    labels = [
        pl.DataFrame({"i": [0, 1]}),
        pl.DataFrame({"j": [0, 1, 2]}),
        pl.DataFrame({"k": [0, 1, 2, 3]}),
    ]

    array1 = create_array(values1, labels, implementation)
    array2 = create_array(values2, labels, implementation)

    # Align axes 0 and 1 only (not 2)
    result1, result2 = align(array1, array2, axes=[(0, 0), (1, 1)])

    assert_array_shape_equal(result1, (2, 3, 4))
    assert_array_shape_equal(result2, (2, 3, 4))

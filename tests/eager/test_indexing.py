import numpy as np
import polars as pl

import polder as pld
from polder.eager.labels import Labels

from ..utils import generate_random_array


def test_indexing_slice(value_array_type):
    """Test indexing with a slice."""
    array = generate_random_array(value_array_type, shape=(5, 3))
    indexed = array[1:3]
    assert indexed.shape() == (2, 3)
    assert isinstance(indexed.values(), value_array_type.value)


def test_indexing_multiple_axes(value_array_type):
    """Test indexing multiple axes at once."""
    array = generate_random_array(value_array_type, shape=(4, 5))
    indexed = array[[1, 3], [0, 2, 4]]
    assert indexed.shape() == (2, 3)
    assert isinstance(indexed.values(), value_array_type.value)


# ============================================================================
# Tests for integer indexing
# ============================================================================


def test_integer_indexing_reduces_dimensionality_2d(value_array_type):
    """Test that integer indexing reduces dimensionality from 2D to 1D.

    Indexing a 2D array with a single integer on the first axis should
    return a 1D array with the second axis preserved.
    """
    # Create (5, 3) array
    values = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0],
        ]
    )
    labels = [pl.DataFrame({"x": [0, 1, 2, 3, 4]}), pl.DataFrame({"y": [10, 20, 30]})]
    array = pld.from_values_and_labels(values, labels)

    # Index with integer on first axis
    result = array[2]
    assert result.shape() == (3,)
    assert Labels(result.labels()) == Labels([array.labels(1)])
    np.testing.assert_array_almost_equal(result.values(), values[2])

    # Index with integer on second axis
    result = array[:, 1]
    assert result.shape() == (5,)
    assert Labels(result.labels()) == Labels([array.labels(0)])
    np.testing.assert_array_almost_equal(result.values(), values[:, 1])


def test_integer_indexing_3d_preserves_remaining_axes(value_array_type):
    """Test that indexing a 3D array with an integer preserves the remaining 2 axes."""
    # Create (2, 3, 4) array
    np.random.seed(789)
    values = np.random.randn(2, 3, 4)
    labels = [
        pl.DataFrame({"i": [0, 1]}),
        pl.DataFrame({"j": [0, 1, 2]}),
        pl.DataFrame({"k": [0, 1, 2, 3]}),
    ]
    array = pld.from_values_and_labels(values, labels)

    # Index first axis
    result = array[1]
    assert result.shape() == (3, 4)
    assert Labels(result.labels()) == Labels(array.labels()[1:])
    np.testing.assert_array_almost_equal(result.values(), values[1])

    # Index with slice and then integer
    result = array[:, :, 2]
    assert result.shape() == (2, 3)
    assert Labels(result.labels()) == Labels(array.labels()[:2])
    np.testing.assert_array_almost_equal(result.values(), values[:, :, 2])


# ============================================================================
# Tests for mapping indexing
# ============================================================================


def test_mapping_indexing_reduces_dimensionality(value_array_type):
    """Test that mapping indexing can reduce dimensionality.

    When filtering with a mapping reduces a dimension to a single value,
    that dimension should be removed from the result.
    """
    # Create (3, 4) array
    values = np.array(
        [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
        ]
    )
    labels = [
        pl.DataFrame({"x": ["a", "b", "c"]}),
        pl.DataFrame({"y": [10, 20, 30, 40]}),
    ]
    array = pld.from_values_and_labels(values, labels)

    # Filter first axis to a single value
    result = array[{"x": "b"}, :]
    assert result.shape() == (4,)
    assert Labels(result.labels()) == Labels(array.labels()[1:])
    np.testing.assert_array_almost_equal(result.values(), values[1])

    # Filter second axis to a single value
    result = array[:, dict(y=20)]
    assert result.shape() == (3,)
    assert Labels(result.labels()) == Labels(array.labels()[:1])
    np.testing.assert_array_almost_equal(result.values(), values[:, 1])


def test_mapping_indexing_preserves_other_dimensions(value_array_type):
    """Test that filtering one dimension with mapping preserves others."""
    # Create (2, 3, 4) array
    values = np.random.randn(2, 3, 4)
    labels = [
        pl.DataFrame({"i": [0, 1]}),
        pl.DataFrame({"j": ["x", "y", "z"]}),
        pl.DataFrame({"k": [10, 20, 30, 40]}),
    ]
    array = pld.from_values_and_labels(values, labels)

    # Filter middle dimension to single value
    result = array[:, {"j": "y"}, :]
    assert result.shape() == (2, 4)
    assert Labels(result.labels()) == Labels([array.labels(0), array.labels(-1)])
    np.testing.assert_array_almost_equal(result.values(), values[:, 1, :])


def test_mapping_indexing_multiple_columns_partial(value_array_type):
    """Test that mapping indexing with multiple columns filters based on available columns."""
    # Create array with multi-column labels
    values = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ]
    )
    labels = [
        pl.DataFrame(
            {
                "letter": ["a", "b", "c"],
                "number": [1, 2, 3],
            }
        ),
        pl.DataFrame({"y": [10, 20, 30]}),
    ]
    array = pld.from_values_and_labels(values, labels)

    # Filter using one column filters the first dimension but keeps letter column
    result = array[{"number": 2}, :]
    # Result should be (1, 3) since we filtered to one row but didn't remove the axis.
    # The letter column is preserved (containing just "b").
    assert result.shape() == (1, 3)
    result_labels_0 = result.labels(0)
    assert (
        result_labels_0 is not None
        and result_labels_0.columns == ["letter"]
        and result_labels_0.item() == "b"
    )
    np.testing.assert_array_almost_equal(result.values(), values[1:2, :])

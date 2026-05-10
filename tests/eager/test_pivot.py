import numpy as np
import polars as pl
import pytest

import polder as pld

from ..utils import cast_value_array


def test_pivot_single_group(value_array_type):
    """Test basic pivot with a single group of columns."""

    # Create array with shape (8, 3) labeled by ["x", "y", "t"] for axis 0.
    values = cast_value_array(
        np.arange(8 * 3).reshape(8, 3).astype(float), value_array_type
    )
    labels = [
        pl.DataFrame(
            {
                "x": [0, 0, 0, 0, 1, 1, 1, 1],
                "y": [0, 0, 1, 1, 0, 0, 1, 1],
                "t": [0, 1, 0, 1, 0, 1, 0, 1],
            }
        ),
        pl.DataFrame({"extra": [0, 1, 2]}),
    ]
    arr = pld.from_values_and_labels(values, labels)

    # Pivot axis 0, moving "t" to a new axis.
    arr_pivoted = arr.pivot(axis_labels_to_pivot={0: ["t"]})
    assert isinstance(arr_pivoted.values(), value_array_type.value)

    # Should create shape (4, 2, 3): 4 unique (x, y) pairs, 2 t labels, 3 extra labels.
    assert arr_pivoted.shape() == (4, 2, 3)

    # Check that the labels are correct.
    keep_labels, pivot_labels, extra_labels = arr_pivoted.labels()

    # 4 rows, 2 columns (x, y):
    assert keep_labels is not None and keep_labels.shape == (4, 2)
    # 2 rows, 1 column (t):
    assert pivot_labels is not None and pivot_labels.shape == (2, 1)
    # 3 rows, 1 column (extra):
    assert extra_labels is not None and extra_labels.shape == (3, 1)

    # Since all labels are sorted in "row-major" order (`t` changes fastest), the pivoting should
    # amount to just a reshape. In other words, the flattened values array should remain equal.
    np.testing.assert_array_equal(
        arr.values().flatten(), arr_pivoted.values().flatten()
    )

    # Check that specifying all labels gives the same result. Note that we don't need to align here,
    # since we maintain order when determining the unique labels. In a non-eager implementation this
    # order may not be guaranteed.
    arr_pivoted_2 = arr.pivot(
        axis_labels_to_pivot={0: [["x", "y"], ["t"]], 1: ["extra"]}
    )
    assert arr_pivoted_2.equals(arr_pivoted)


def test_pivot_multiple_groups(value_array_type):
    """Test pivot with multiple groups creating multiple new axes."""

    # Create array with 16 rows labeled by ["x", "y", "t", "s"] and 2 extra labels.
    values = cast_value_array(
        np.arange(16 * 2).reshape(16, 2).astype(float), value_array_type
    )
    labels = [
        pl.DataFrame(
            {
                "x": [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                "y": [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1],
                "t": [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
                "s": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            }
        ),
        pl.DataFrame({"extra": [0, 1]}),
    ]
    arr = pld.from_values_and_labels(values, labels)

    # Pivot axis 0, separating into [["t"], ["s"]]
    arr_pivoted = arr.pivot(axis_labels_to_pivot={0: [["t"], ["s"]]})
    assert isinstance(arr_pivoted.values(), value_array_type.value)

    # Should create shape (4, 2, 2, 2):
    # - 4 unique (x, y) pairs
    # - 2 t values -> first new axis
    # - 2 s values -> second new axis
    # - 2 columns
    assert arr_pivoted.shape() == (4, 2, 2, 2)

    keep_labels, t_labels, s_labels, extra_labels = arr_pivoted.labels()

    assert keep_labels is not None and keep_labels.shape == (4, 2)
    assert t_labels is not None and t_labels.shape == (2, 1)
    assert s_labels is not None and s_labels.shape == (2, 1)
    assert extra_labels is not None and extra_labels.shape == (2, 1)

    # Check that specifying all labels gives the same result.
    arr_pivoted_2 = arr.pivot(
        axis_labels_to_pivot={0: [["x", "y"], ["t"], ["s"]], 1: ["extra"]}
    )
    assert arr_pivoted_2.equals(arr_pivoted)


def test_pivot_with_fill_value(value_array_type):
    """Test pivot with missing combinations and fill_value."""

    # Create incomplete data where not all (x, y, t) combinations exist.
    values = cast_value_array(
        np.arange(4 * 2, dtype=float).reshape((4, 2)), value_array_type
    )
    labels = [
        pl.DataFrame(
            {
                "x": [0, 0, 1, 1],
                "y": [0, 1, 0, 1],
                "t": [0, 0, 1, 1],
            }
        ),
        pl.DataFrame({"extra": [0, 1]}),
    ]
    arr = pld.from_values_and_labels(values, labels)

    # Pivot with fill_value for missing t=1 combinations.
    arr_pivoted = arr.pivot(axis_labels_to_pivot={0: ["t"]}, fill_value=np.nan)
    assert isinstance(arr_pivoted.values(), value_array_type.value)

    # Should create shape (4, 2, 2):
    # - 4 unique (x, y) pairs
    # - 2 t labels
    # - 2 extra labels
    # Note that we now have 16 values instead of 8.
    assert arr_pivoted.shape() == (4, 2, 2)

    # Half of the values should be NaN, since we only had t=0 for half of them and only t=1 for the
    # other half.
    assert np.sum(np.isnan(arr_pivoted.values())) == 8

    # Check that specifying all labels gives the same result.
    arr_pivoted_2 = arr.pivot(
        axis_labels_to_pivot={0: [["x", "y"], ["t"]], 1: ["extra"]}, fill_value=np.nan
    )
    assert arr_pivoted_2.equals(arr_pivoted)

    # Check that pivoting without fill value raises.
    with pytest.raises(match="no `fill_value` provided"):
        arr.pivot(axis_labels_to_pivot={0: ["t"]})


def test_unpivot_single_axis(value_array_type):
    """Test unpivot with a single axis merge."""

    # Create array with shape (4, 2, 3) from the first test_pivot_single_group test.
    values = cast_value_array(
        np.arange(8 * 3).reshape(8, 3).astype(float), value_array_type
    )
    labels = [
        pl.DataFrame(
            {
                "x": [0, 0, 0, 0, 1, 1, 1, 1],
                "y": [0, 0, 1, 1, 0, 0, 1, 1],
                "t": [0, 1, 0, 1, 0, 1, 0, 1],
            }
        ),
        pl.DataFrame({"extra": [0, 1, 2]}),
    ]
    arr = pld.from_values_and_labels(values, labels)

    # Pivot to create shape (4, 2, 3).
    arr_pivoted = arr.pivot(axis_labels_to_pivot={0: ["t"]})
    assert arr_pivoted.shape() == (4, 2, 3)
    assert isinstance(arr_pivoted.values(), value_array_type.value)

    # Unpivot back to shape (8, 3).
    arr_unpivoted = arr_pivoted.unpivot(axes_to_merge=[(0, 1)])

    # Should restore original shape.
    assert arr_unpivoted.shape() == (8, 3)
    assert isinstance(arr_unpivoted.values(), value_array_type.value)

    # Values should be the same (just reshaped).
    np.testing.assert_array_equal(arr.values(), arr_unpivoted.values())

    # The unpivoted array should equal the original.
    assert arr.equals(arr_unpivoted)


def test_unpivot_multiple_axes(value_array_type):
    """Test unpivot with multiple separate merges."""

    # Create array with 4 x 2 x 2 x 2 shape.
    values = cast_value_array(
        np.arange(4 * 2 * 2 * 2).reshape(4, 2, 2, 2).astype(float), value_array_type
    )
    labels = [
        pl.DataFrame({"x": [0, 0, 1, 1], "y": [0, 1, 0, 1]}),
        pl.DataFrame({"t": [0, 1]}),
        pl.DataFrame({"s": [0, 1]}),
        pl.DataFrame({"extra": [0, 1]}),
    ]
    arr = pld.from_values_and_labels(values, labels)

    # Unpivot (0, 1) to merge (x, y) with (t).
    arr_unpivoted_1 = arr.unpivot(axes_to_merge=[(0, 1)])

    # Should have shape (8, 2, 2): 4*2=8 from merging (x,y) with (t), then (s) and (extra).
    assert arr_unpivoted_1.shape() == (8, 2, 2)
    assert isinstance(arr_unpivoted_1.values(), value_array_type.value)

    # Then unpivot (1, 2) to merge (s) with (extra).
    arr_unpivoted_2 = arr_unpivoted_1.unpivot(axes_to_merge=[(1, 2)])

    # Should have shape (8, 4): keeping the 8 from first merge, 2*2=4 from second.
    assert arr_unpivoted_2.shape() == (8, 4)
    assert isinstance(arr_unpivoted_2.values(), value_array_type.value)

    # Values should match the original (reshaped).
    np.testing.assert_array_equal(arr.values().reshape(8, 4), arr_unpivoted_2.values())


def test_unpivot_overlapping_axes_error(value_array_type):
    """Test that overlapping axes raise an error."""

    # Create array with shape (4, 2, 2, 2) with labeled axes.
    values = cast_value_array(
        np.arange(4 * 2 * 2 * 2).reshape(4, 2, 2, 2).astype(float), value_array_type
    )
    labels = [
        pl.DataFrame({"a": [0, 0, 1, 1], "b": [0, 1, 0, 1]}),
        pl.DataFrame({"c": [0, 1]}),
        pl.DataFrame({"d": [0, 1]}),
        pl.DataFrame({"e": [0, 1]}),
    ]
    arr = pld.from_values_and_labels(values, labels)

    # Axes (0, 2) and (1, 3) overlap.
    with pytest.raises(match="Axis slices are invalid"):
        arr.unpivot(axes_to_merge=[(0, 2), (1, 3)])


def test_unpivot_unordered_axes_error(value_array_type):
    """Test that unordered axes raise an error."""

    # Create array with shape (2, 2, 2, 2) with labeled axes.
    values = cast_value_array(
        np.arange(2 * 2 * 2 * 2).reshape(2, 2, 2, 2).astype(float), value_array_type
    )
    labels = [
        pl.DataFrame({"a": [0, 1]}),
        pl.DataFrame({"b": [0, 1]}),
        pl.DataFrame({"c": [0, 1]}),
        pl.DataFrame({"d": [0, 1]}),
    ]
    arr = pld.from_values_and_labels(values, labels)

    # Axes (2, 3) and (0, 1) are not in order.
    with pytest.raises(match="Axis slices are invalid"):
        arr.unpivot(axes_to_merge=[(2, 3), (0, 1)])


def test_unpivot_single_valued_slice_error(value_array_type):
    """Test that single-valued slices raise an error."""

    # Create array with shape (2, 2, 2) with labeled axes.
    values = cast_value_array(
        np.arange(2 * 2 * 2).reshape(2, 2, 2).astype(float), value_array_type
    )
    labels = [
        pl.DataFrame({"a": [0, 1]}),
        pl.DataFrame({"b": [0, 1]}),
        pl.DataFrame({"c": [0, 1]}),
    ]
    arr = pld.from_values_and_labels(values, labels)

    # (1, 1) is a single-valued slice.
    with pytest.raises(match="Axis slices are invalid"):
        arr.unpivot(axes_to_merge=[(1, 1)])


def test_unpivot_three_axes_merge(value_array_type):
    """Test unpivot with three consecutive axes."""

    # Create array with shape (2, 2, 2, 3) with labeled axes.
    values = cast_value_array(
        np.arange(2 * 2 * 2 * 3).reshape(2, 2, 2, 3).astype(float), value_array_type
    )
    labels = [
        pl.DataFrame({"a": [0, 1]}),
        pl.DataFrame({"b": [0, 1]}),
        pl.DataFrame({"c": [0, 1]}),
        pl.DataFrame({"d": [0, 1, 2]}),
    ]
    arr = pld.from_values_and_labels(values, labels)

    # Unpivot axes (0, 1, 2) - three axes merged into one.
    arr_unpivoted = arr.unpivot(axes_to_merge=[(0, 2)])

    # Should have shape (8, 3): 2*2*2=8 from merging three axes, 3 from the last axis.
    assert arr_unpivoted.shape() == (8, 3)
    assert isinstance(arr_unpivoted.values(), value_array_type.value)

    # Values should be reshaped.
    expected_values = values.reshape(8, 3)
    np.testing.assert_array_equal(arr_unpivoted.values(), expected_values)

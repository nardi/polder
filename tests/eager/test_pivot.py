import numpy as np
import polars as pl
import pytest

import polder as pld


def test_pivot_single_group():
    """Test basic pivot with a single group of columns."""

    # Create array with shape (8, 3) labeled by ["x", "y", "t"] for axis 0.
    values = np.arange(8 * 3).reshape(8, 3).astype(float)
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


def test_pivot_multiple_groups():
    """Test pivot with multiple groups creating multiple new axes."""

    # Create array with 16 rows labeled by ["x", "y", "t", "s"] and 2 extra labels.
    values = np.arange(16 * 2).reshape(16, 2).astype(float)
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


def test_pivot_with_fill_value():
    """Test pivot with missing combinations and fill_value."""

    # Create incomplete data where not all (x, y, t) combinations exist.
    values = np.arange(4 * 2, dtype=float).reshape((4, 2))
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

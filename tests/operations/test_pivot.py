"""Tests for pivot/unpivot operations dispatched over both eager and lazy arrays."""

import numpy as np
import polars as pl
import pytest

import polder as pld
from polder.eager.labels import Labels
from polder.protocols.implementations import EAGER, LAZY

from ..utils import ValueArrayType, cast_value_array


@pytest.fixture(params=[EAGER, LAZY])
def implementation(request):
    """Parametrized fixture for testing both eager and lazy implementations."""
    return request.param


def _make_array(values, labels, value_array_type, implementation):
    """Helper: cast values and create array for the given implementation."""
    return pld.from_values_and_labels(
        cast_value_array(values, value_array_type),
        labels,
        implementation=implementation,
    )


def _assert_values_equal(result, expected_np):
    """Compare result.values() (always numpy for lazy, typed for eager) to expected numpy."""
    np.testing.assert_array_equal(result.values(), expected_np)


def _assert_values_almost_equal(result, expected_np):
    np.testing.assert_array_almost_equal(result.values(), expected_np)


# ---------------------------------------------------------------------------
# pivot tests
# ---------------------------------------------------------------------------


def test_pivot_single_group(value_array_type, implementation):
    """Basic pivot: split one labeled axis into keep + pivot axes."""
    if implementation == LAZY and value_array_type == ValueArrayType.JAX:
        pytest.skip("Lazy arrays do not support JAX value arrays.")

    values = np.arange(8 * 3).reshape(8, 3).astype(float)
    labels = [
        pl.DataFrame({
            "x": [0, 0, 0, 0, 1, 1, 1, 1],
            "y": [0, 0, 1, 1, 0, 0, 1, 1],
            "t": [0, 1, 0, 1, 0, 1, 0, 1],
        }),
        pl.DataFrame({"extra": [0, 1, 2]}),
    ]
    arr = _make_array(values, labels, value_array_type, implementation)
    arr_pivoted = arr.pivot(axis_labels_to_pivot={0: ["t"]})

    assert arr_pivoted.shape() == (4, 2, 3)

    keep_labels, pivot_labels, extra_labels = arr_pivoted.labels()
    assert keep_labels is not None and keep_labels.shape == (4, 2)
    assert pivot_labels is not None and pivot_labels.shape == (2, 1)
    assert extra_labels is not None and extra_labels.shape == (3, 1)

    # Since labels are in row-major order, pivot is equivalent to a reshape.
    _assert_values_equal(arr, values)
    np.testing.assert_array_equal(
        arr.values().flatten(), arr_pivoted.values().flatten()
    )


def test_pivot_multiple_groups(value_array_type, implementation):
    """Pivot splits axis into three axes via multiple groups."""
    if implementation == LAZY and value_array_type == ValueArrayType.JAX:
        pytest.skip("Lazy arrays do not support JAX value arrays.")

    values = np.arange(16 * 2).reshape(16, 2).astype(float)
    labels = [
        pl.DataFrame({
            "x": [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
            "y": [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1],
            "t": [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
            "s": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        }),
        pl.DataFrame({"extra": [0, 1]}),
    ]
    arr = _make_array(values, labels, value_array_type, implementation)
    arr_pivoted = arr.pivot(axis_labels_to_pivot={0: [["t"], ["s"]]})

    assert arr_pivoted.shape() == (4, 2, 2, 2)

    keep_labels, t_labels, s_labels, extra_labels = arr_pivoted.labels()
    assert keep_labels is not None and keep_labels.shape == (4, 2)
    assert t_labels is not None and t_labels.shape == (2, 1)
    assert s_labels is not None and s_labels.shape == (2, 1)
    assert extra_labels is not None and extra_labels.shape == (2, 1)


def test_pivot_with_fill_value(value_array_type, implementation):
    """Pivot with missing combinations fills them with the provided fill_value."""
    if implementation == LAZY and value_array_type == ValueArrayType.JAX:
        pytest.skip("Lazy arrays do not support JAX value arrays.")

    values = np.arange(4 * 2, dtype=float).reshape((4, 2))
    labels = [
        pl.DataFrame({
            "x": [0, 0, 1, 1],
            "y": [0, 1, 0, 1],
            "t": [0, 0, 1, 1],
        }),
        pl.DataFrame({"extra": [0, 1]}),
    ]
    arr = _make_array(values, labels, value_array_type, implementation)

    arr_pivoted = arr.pivot(axis_labels_to_pivot={0: ["t"]}, fill_value=np.nan)
    assert arr_pivoted.shape() == (4, 2, 2)
    assert np.sum(np.isnan(arr_pivoted.values())) == 8

    # Without fill_value, the eager implementation raises at pivot() time;
    # the lazy implementation defers the error until values() is called.
    if implementation == EAGER:
        with pytest.raises(Exception, match="no `fill_value` provided"):
            arr.pivot(axis_labels_to_pivot={0: ["t"]})
    else:
        arr_no_fill = arr.pivot(axis_labels_to_pivot={0: ["t"]})
        with pytest.raises(Exception):
            arr_no_fill.values()


# ---------------------------------------------------------------------------
# unpivot tests
# ---------------------------------------------------------------------------


def test_unpivot_single_axis(value_array_type, implementation):
    """Unpivot restores a previously pivoted array."""
    if implementation == LAZY and value_array_type == ValueArrayType.JAX:
        pytest.skip("Lazy arrays do not support JAX value arrays.")

    values = np.arange(8 * 3).reshape(8, 3).astype(float)
    labels = [
        pl.DataFrame({
            "x": [0, 0, 0, 0, 1, 1, 1, 1],
            "y": [0, 0, 1, 1, 0, 0, 1, 1],
            "t": [0, 1, 0, 1, 0, 1, 0, 1],
        }),
        pl.DataFrame({"extra": [0, 1, 2]}),
    ]
    arr = _make_array(values, labels, value_array_type, implementation)
    arr_pivoted = arr.pivot(axis_labels_to_pivot={0: ["t"]})
    assert arr_pivoted.shape() == (4, 2, 3)

    arr_unpivoted = arr_pivoted.unpivot(axes_to_merge=[(0, 1)])
    assert arr_unpivoted.shape() == (8, 3)

    np.testing.assert_array_equal(arr.values(), arr_unpivoted.values())
    assert Labels(arr_unpivoted.labels()) == Labels(arr.labels())


def test_unpivot_multiple_axes(value_array_type, implementation):
    """Unpivot with two separate merge groups."""
    if implementation == LAZY and value_array_type == ValueArrayType.JAX:
        pytest.skip("Lazy arrays do not support JAX value arrays.")

    values = np.arange(4 * 2 * 2 * 2).reshape(4, 2, 2, 2).astype(float)
    labels = [
        pl.DataFrame({"x": [0, 0, 1, 1], "y": [0, 1, 0, 1]}),
        pl.DataFrame({"t": [0, 1]}),
        pl.DataFrame({"s": [0, 1]}),
        pl.DataFrame({"extra": [0, 1]}),
    ]
    arr = _make_array(values, labels, value_array_type, implementation)

    arr_unpivoted_1 = arr.unpivot(axes_to_merge=[(0, 1)])
    assert arr_unpivoted_1.shape() == (8, 2, 2)

    arr_unpivoted_2 = arr_unpivoted_1.unpivot(axes_to_merge=[(1, 2)])
    assert arr_unpivoted_2.shape() == (8, 4)

    np.testing.assert_array_equal(arr.values().reshape(8, 4), arr_unpivoted_2.values())


def test_unpivot_overlapping_axes_error(value_array_type, implementation):
    """Overlapping axis slices raise an error."""
    if implementation == LAZY and value_array_type == ValueArrayType.JAX:
        pytest.skip("Lazy arrays do not support JAX value arrays.")

    values = np.arange(4 * 2 * 2 * 2).reshape(4, 2, 2, 2).astype(float)
    labels = [
        pl.DataFrame({"a": [0, 0, 1, 1], "b": [0, 1, 0, 1]}),
        pl.DataFrame({"c": [0, 1]}),
        pl.DataFrame({"d": [0, 1]}),
        pl.DataFrame({"e": [0, 1]}),
    ]
    arr = _make_array(values, labels, value_array_type, implementation)
    with pytest.raises(Exception, match="Axis slices are invalid"):
        arr.unpivot(axes_to_merge=[(0, 2), (1, 3)])


def test_unpivot_unordered_axes_error(value_array_type, implementation):
    """Unordered axis slices raise an error."""
    if implementation == LAZY and value_array_type == ValueArrayType.JAX:
        pytest.skip("Lazy arrays do not support JAX value arrays.")

    values = np.arange(2 * 2 * 2 * 2).reshape(2, 2, 2, 2).astype(float)
    labels = [
        pl.DataFrame({"a": [0, 1]}),
        pl.DataFrame({"b": [0, 1]}),
        pl.DataFrame({"c": [0, 1]}),
        pl.DataFrame({"d": [0, 1]}),
    ]
    arr = _make_array(values, labels, value_array_type, implementation)
    with pytest.raises(Exception, match="Axis slices are invalid"):
        arr.unpivot(axes_to_merge=[(2, 3), (0, 1)])


def test_unpivot_single_valued_slice_error(value_array_type, implementation):
    """Single-valued slices raise an error."""
    if implementation == LAZY and value_array_type == ValueArrayType.JAX:
        pytest.skip("Lazy arrays do not support JAX value arrays.")

    values = np.arange(2 * 2 * 2).reshape(2, 2, 2).astype(float)
    labels = [
        pl.DataFrame({"a": [0, 1]}),
        pl.DataFrame({"b": [0, 1]}),
        pl.DataFrame({"c": [0, 1]}),
    ]
    arr = _make_array(values, labels, value_array_type, implementation)
    with pytest.raises(Exception, match="Axis slices are invalid"):
        arr.unpivot(axes_to_merge=[(1, 1)])


def test_unpivot_three_axes_merge(value_array_type, implementation):
    """Three consecutive axes can be merged into one."""
    if implementation == LAZY and value_array_type == ValueArrayType.JAX:
        pytest.skip("Lazy arrays do not support JAX value arrays.")

    values = np.arange(2 * 2 * 2 * 3).reshape(2, 2, 2, 3).astype(float)
    labels = [
        pl.DataFrame({"a": [0, 1]}),
        pl.DataFrame({"b": [0, 1]}),
        pl.DataFrame({"c": [0, 1]}),
        pl.DataFrame({"d": [0, 1, 2]}),
    ]
    arr = _make_array(values, labels, value_array_type, implementation)

    arr_unpivoted = arr.unpivot(axes_to_merge=[(0, 2)])
    assert arr_unpivoted.shape() == (8, 3)

    np.testing.assert_array_equal(arr_unpivoted.values(), values.reshape(8, 3))

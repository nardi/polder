"""Tests for LazyFrameLabeledArray creation functions."""

import narwhals as nw
import numpy as np
import polars as pl

import polder as pld
from polder.protocols.implementations import LAZY


def test_array_creation_from_values_and_labels():
    """Test creating a lazy array from values and labels."""
    values = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    labels = [
        pl.DataFrame({"x": [0, 1]}),
        pl.DataFrame({"y": [0, 1, 2]}),
    ]

    array = pld.from_values_and_labels(values, labels, implementation=LAZY)

    assert array.shape() == (2, 3)
    np.testing.assert_array_equal(array.values(), values)
    assert len(array.labels()) == 2


def test_array_creation_with_different_shapes():
    """Test creating lazy arrays with different shapes."""
    for shape in [(1, 1), (3, 4), (5, 2, 3)]:
        np.random.seed(42)
        values = np.random.randn(*shape)
        labels = [pl.DataFrame({"dim0": np.arange(size)}) for size in shape]

        array = pld.from_values_and_labels(values, labels, implementation=LAZY)

        assert array.shape() == shape
        assert array.values().shape == shape


def test_array_values_preservation():
    """Test that values are correctly preserved through lazy evaluation."""
    values = np.array([[1.5, 2.5], [3.5, 4.5], [5.5, 6.5]])
    labels = [
        pl.DataFrame({"x": [10, 20, 30]}),
        pl.DataFrame({"y": [100, 200]}),
    ]

    array = pld.from_values_and_labels(values, labels, implementation=LAZY)

    result_values = array.values()
    assert isinstance(result_values, np.ndarray)
    assert result_values.shape == (3, 2)
    np.testing.assert_array_almost_equal(result_values, values)


def test_array_labels_preservation():
    """Test accessing and preserving labels."""
    values = np.array([[1.0, 2.0], [3.0, 4.0]])
    labels = [
        pl.DataFrame({"x": [10, 20]}),
        pl.DataFrame({"y": [100, 200]}),
    ]

    array = pld.from_values_and_labels(values, labels, implementation=LAZY)

    labels0, labels1 = array.labels()

    # Check that label content is preserved.
    assert labels0 is not None
    assert labels0.to_native().equals(labels[0])
    assert labels1 is not None
    assert labels1.to_native().equals(labels[1])


def test_array_labels_with_multiple_columns():
    """Test labels with multiple columns."""
    values = np.array([[1.0, 2.0], [3.0, 4.0]])
    labels = [
        pl.DataFrame({"x_idx": [0, 1], "x_name": ["a", "b"]}),
        pl.DataFrame({"y_idx": [0, 1], "y_name": ["c", "d"]}),
    ]

    array = pld.from_values_and_labels(values, labels, implementation=LAZY)

    labels0, labels1 = array.labels()
    assert labels0 is not None
    assert type(labels0.to_native()) is type(labels[0])
    assert labels0.shape == (2, 2)
    assert labels1 is not None
    assert type(labels1.to_native()) is type(labels[1])
    assert labels1.shape == (2, 2)


def test_array_single_axis():
    """Test creating a lazy array with a single axis."""
    values = np.array([1.0, 2.0, 3.0, 4.0])
    labels = [
        pl.DataFrame({"idx": [0, 1, 2, 3]}),
    ]

    array = pld.from_values_and_labels(values, labels, implementation=LAZY)

    assert array.shape() == (4,)
    np.testing.assert_array_equal(array.values(), values)

    (label_frame,) = array.labels()
    assert label_frame is not None
    assert type(label_frame.to_native()) is type(labels[0])
    assert label_frame.shape == (4, 1)


def test_array_three_dimensions():
    """Test creating a lazy array with three dimensions."""
    shape = (2, 3, 4)
    values = np.arange(np.prod(shape)).reshape(shape).astype(float)
    labels = [
        pl.DataFrame({"x": [0, 1]}),
        pl.DataFrame({"y": [0, 1, 2]}),
        pl.DataFrame({"z": [0, 1, 2, 3]}),
    ]

    array = pld.from_values_and_labels(values, labels, implementation=LAZY)

    assert array.shape() == shape
    np.testing.assert_array_equal(array.values(), values)

    array_labels = array.labels()
    assert len(array_labels) == 3
    for l1, l2 in zip(labels, array_labels):
        assert l2 is not None
        assert type(l2.to_native()) is type(l1)
        assert l2.shape == l1.shape


def test_from_frame_creation():
    """Test creating a lazy array from a frame with value column."""
    frame_data = {
        "val": [1.0, 2.0, 3.0, 4.0],
        "label_a": ["x", "y", "z", "w"],
    }
    frame = pl.DataFrame(frame_data)

    array = pld.from_frame(
        nw.from_native(frame), value_column="val", implementation=LAZY
    )

    assert array.shape() == (4,)
    np.testing.assert_array_equal(array.values(), np.array([1.0, 2.0, 3.0, 4.0]))

    (label_frame,) = array.labels()
    assert label_frame is not None
    assert type(label_frame.to_native()) is type(frame)
    assert label_frame.shape == (4, 1)


def test_from_frame_with_multiple_label_columns():
    """Test creating a lazy array from a frame with multiple label columns."""
    frame_data = {
        "value": [10.0, 20.0, 30.0, 50.0],
        "idx": [0, 1, 2, 4],
        "name": ["first", "second", "third", "q"],
        "group": ["a", "a", "b", "c"],
    }
    frame = pl.DataFrame(frame_data)

    array = pld.from_frame(
        nw.from_native(frame), value_column="value", implementation=LAZY
    )

    assert array.shape() == (4,)

    (label_frame,) = array.labels()
    assert label_frame is not None
    assert type(label_frame.to_native()) is type(frame)
    # Labels should contain all columns except the value column, so 3 columns of length 4.
    assert label_frame.shape == (4, 3)


def test_values_returns_numpy_array():
    """Test that values() always returns numpy arrays regardless of backend."""
    import jax.numpy as jnp

    values = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    labels = [
        pl.DataFrame({"x": [0, 1]}),
        pl.DataFrame({"y": [0, 1]}),
    ]

    array = pld.from_values_and_labels(values, labels, implementation=LAZY)

    result = array.values()
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float32


def test_shape_method():
    """Test the shape() method returns correct shape."""
    for shape in [(5,), (2, 3), (2, 3, 4), (1, 1, 1, 1)]:
        values = np.arange(np.prod(shape)).reshape(shape).astype(float)
        labels = [pl.DataFrame({f"axis{i}": np.arange(s)}) for i, s in enumerate(shape)]

        array = pld.from_values_and_labels(values, labels, implementation=LAZY)

        assert array.shape() == shape


def test_integer_values():
    """Test arrays with integer values."""
    values = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int32)
    labels = [
        pl.DataFrame({"x": [0, 1, 2]}),
        pl.DataFrame({"y": [0, 1]}),
    ]

    array = pld.from_values_and_labels(values, labels, implementation=LAZY)

    result = array.values()
    np.testing.assert_array_equal(result, values)


def test_nan_values():
    """Test arrays containing NaN values."""
    values = np.array([[1.0, np.nan], [np.inf, -np.inf]])
    labels = [
        pl.DataFrame({"x": [0, 1]}),
        pl.DataFrame({"y": [0, 1]}),
    ]

    array = pld.from_values_and_labels(values, labels, implementation=LAZY)

    result = array.values()
    # Use array_equal with equal_nan since NaN != NaN.
    np.testing.assert_equal(
        np.isnan(result) | (result == values),
        np.ones_like(result, dtype=bool),
    )

"""Tests for pld.convert(), which converts arrays between eager and lazy implementations."""

import narwhals as nw
import numpy as np
import polars as pl
import pytest

import polder as pld
from polder.eager.array import EagerFrameLabeledArray
from polder.lazy.array import LazyFrameLabeledArray
from polder.protocols.implementations import EAGER, LAZY

# Shared test data.
_VALUES = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
_LABELS = [
    pl.DataFrame({"x": [10, 20]}),
    pl.DataFrame({"y": [100, 200, 300]}),
]


@pytest.fixture
def eager_array():
    """An eager array used as the source for convert(..., implementation=LAZY) tests."""
    return pld.from_values_and_labels(_VALUES, _LABELS, implementation=EAGER)


@pytest.fixture
def lazy_array():
    """A lazy array used as the source for convert(..., implementation=EAGER) tests."""
    return pld.from_values_and_labels(_VALUES, _LABELS, implementation=LAZY)


@pytest.fixture
def eager_to_lazy(eager_array):
    """The result of converting an eager array to lazy."""
    return pld.convert(eager_array, implementation=LAZY)


@pytest.fixture
def lazy_to_eager(lazy_array):
    """The result of converting a lazy array to eager."""
    return pld.convert(lazy_array, implementation=EAGER)


# convert(..., implementation=LAZY) tests


def test_to_lazy_returns_lazy_array(eager_to_lazy):
    """convert(..., implementation=LAZY) should return a LazyFrameLabeledArray."""
    assert isinstance(eager_to_lazy, LazyFrameLabeledArray)


def test_to_lazy_preserves_values(eager_to_lazy):
    """convert(..., implementation=LAZY) should preserve the array values."""
    np.testing.assert_array_equal(eager_to_lazy.values(), _VALUES)


def test_to_lazy_preserves_labels(eager_to_lazy):
    """convert(..., implementation=LAZY) should preserve the labels for each axis."""
    result_labels = eager_to_lazy.labels()
    for result_label, original_label in zip(result_labels, _LABELS):
        assert result_label is not None
        assert result_label.to_native().equals(original_label)


def test_to_lazy_preserves_shape(eager_to_lazy):
    """convert(..., implementation=LAZY) should preserve the array shape."""
    assert eager_to_lazy.shape() == (2, 3)


def test_to_lazy_internal_frames_are_lazy(eager_array):
    """The result should use LazyFrames internally when eager evaluation is disabled."""
    with pld.config.use_eager_evaluation_for_lazy_arrays(False):
        result = pld.convert(eager_array, implementation=LAZY)

    assert isinstance(result._values, nw.LazyFrame)
    assert isinstance(result._shape, nw.LazyFrame)
    assert all(isinstance(lf, nw.LazyFrame) for lf in result._indexed_labels)


def test_to_lazy_with_none_labels():
    """convert(..., implementation=LAZY) should preserve None (unlabeled) axes.

    A None label is only valid on a size-1 axis.
    """
    values = np.array([[1.0, 2.0, 3.0]])
    labels: list[pl.DataFrame | None] = [None, pl.DataFrame({"y": [0, 1, 2]})]
    array = pld.from_values_and_labels(values, labels, implementation=EAGER)

    result = pld.convert(array, implementation=LAZY)

    assert isinstance(result, LazyFrameLabeledArray)
    none_label, y_label = result.labels()
    assert none_label is None
    assert y_label is not None
    np.testing.assert_array_equal(result.values(), values)


def test_to_lazy_with_multiple_shapes():
    """convert(..., implementation=LAZY) should work for 1D, 2D, and 3D arrays."""
    for shape in [(4,), (2, 3), (2, 3, 4)]:
        values = np.arange(np.prod(shape), dtype=float).reshape(shape)
        labels = [pl.DataFrame({f"axis{i}": np.arange(s)}) for i, s in enumerate(shape)]
        array = pld.from_values_and_labels(values, labels, implementation=EAGER)

        result = pld.convert(array, implementation=LAZY)

        assert result.shape() == shape
        np.testing.assert_array_equal(result.values(), values)


def test_convert_eager_to_eager_returns_same_object(eager_array):
    """convert(..., implementation=EAGER) on an eager array should return it unchanged."""
    result = pld.convert(eager_array, implementation=EAGER)
    assert result is eager_array


# convert(..., implementation=EAGER) tests


def test_to_eager_returns_eager_array(lazy_to_eager):
    """convert(..., implementation=EAGER) should return an EagerFrameLabeledArray."""
    assert isinstance(lazy_to_eager, EagerFrameLabeledArray)


def test_to_eager_preserves_values(lazy_to_eager):
    """convert(..., implementation=EAGER) should preserve the array values."""
    np.testing.assert_array_equal(lazy_to_eager.values(), _VALUES)


def test_to_eager_preserves_labels(lazy_to_eager):
    """convert(..., implementation=EAGER) should preserve the labels for each axis."""
    result_labels = lazy_to_eager.labels()
    for result_label, original_label in zip(result_labels, _LABELS):
        assert result_label is not None
        assert result_label.to_native().equals(original_label)


def test_to_eager_preserves_shape(lazy_to_eager):
    """convert(..., implementation=EAGER) should preserve the array shape."""
    assert lazy_to_eager.shape() == (2, 3)


def test_to_eager_values_are_numpy(lazy_to_eager):
    """convert(..., implementation=EAGER) should return a numpy array from .values()."""
    assert isinstance(lazy_to_eager.values(), np.ndarray)


def test_to_eager_with_none_labels():
    """convert(..., implementation=EAGER) should preserve None (unlabeled) axes.

    A None label is only valid on a size-1 axis.
    """
    values = np.array([[1.0, 2.0, 3.0]])
    labels = [None, pl.DataFrame({"y": [0, 1, 2]})]
    array = pld.from_values_and_labels(values, labels, implementation=LAZY)

    result = pld.convert(array, implementation=EAGER)

    assert isinstance(result, EagerFrameLabeledArray)
    none_label, y_label = result.labels()
    assert none_label is None
    assert y_label is not None
    np.testing.assert_array_equal(result.values(), values)


def test_to_eager_with_multiple_shapes():
    """convert(..., implementation=EAGER) should work for 1D, 2D, and 3D arrays."""
    for shape in [(4,), (2, 3), (2, 3, 4)]:
        values = np.arange(np.prod(shape), dtype=float).reshape(shape)
        labels = [pl.DataFrame({f"axis{i}": np.arange(s)}) for i, s in enumerate(shape)]
        array = pld.from_values_and_labels(values, labels, implementation=LAZY)

        result = pld.convert(array, implementation=EAGER)

        assert result.shape() == shape
        np.testing.assert_array_equal(result.values(), values)


def test_convert_lazy_to_lazy_returns_same_object(lazy_array):
    """convert(..., implementation=LAZY) on a lazy array should return it unchanged."""
    result = pld.convert(lazy_array, implementation=LAZY)
    assert result is lazy_array


# Round-trip tests


def test_round_trip_eager_to_lazy_to_eager(eager_array):
    """Converting eager → lazy → eager should preserve values, labels, and shape."""
    result = pld.convert(
        pld.convert(eager_array, implementation=LAZY), implementation=EAGER
    )

    assert result.shape() == (2, 3)
    np.testing.assert_array_equal(result.values(), _VALUES)
    for result_label, original_label in zip(result.labels(), _LABELS):
        assert result_label is not None
        assert result_label.to_native().equals(original_label)


def test_round_trip_lazy_to_eager_to_lazy(lazy_array):
    """Converting lazy → eager → lazy should preserve values, labels, and shape."""
    result = pld.convert(
        pld.convert(lazy_array, implementation=EAGER), implementation=LAZY
    )

    assert result.shape() == (2, 3)
    np.testing.assert_array_equal(result.values(), _VALUES)
    for result_label, original_label in zip(result.labels(), _LABELS):
        assert result_label is not None
        assert result_label.to_native().equals(original_label)

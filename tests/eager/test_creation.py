import numpy as np

from .utils import generate_random_array


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

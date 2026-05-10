from ..utils import generate_random_array


def test_array_creation(value_array_type):
    """Test creating an array from values and labels."""
    array = generate_random_array(value_array_type)
    assert array.shape() == (2, 3)
    assert array.values().shape == (2, 3)
    assert len(array.labels()) == 2


def test_array_creation_with_different_shapes(value_array_type):
    """Test creating arrays with different shapes."""
    for shape in [(1, 1), (3, 4), (5, 2, 3)]:
        array = generate_random_array(value_array_type, shape=shape)
        assert array.shape() == shape


def test_array_values(value_array_type):
    """Test that values are correctly stored."""
    array = generate_random_array(value_array_type)
    values = array.values()
    assert isinstance(values, value_array_type.value)
    assert values.shape == (2, 3)


def test_array_labels(value_array_type):
    """Test accessing labels."""
    array = generate_random_array(value_array_type)
    labels = array.labels()
    assert len(labels) == 2
    assert all(label is not None for label in labels)

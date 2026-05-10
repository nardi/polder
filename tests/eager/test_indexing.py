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

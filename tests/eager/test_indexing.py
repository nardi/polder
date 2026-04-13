from .utils import generate_random_array


def test_indexing_slice():
    """Test indexing with a slice."""
    array = generate_random_array(shape=(5, 3))
    indexed = array[1:3]
    assert indexed.shape() == (2, 3)


def test_indexing_multiple_axes():
    """Test indexing multiple axes at once."""
    array = generate_random_array(shape=(4, 5))
    indexed = array[[1, 3], [0, 2, 4]]
    assert indexed.shape() == (2, 3)

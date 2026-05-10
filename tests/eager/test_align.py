import pytest

from ..utils import generate_random_array, shuffle_labels


def test_auto_align_default_true():
    """Test that auto_align is True by default."""
    from polder.config import auto_align

    assert auto_align() is True


def test_auto_align_context_manager():
    """Test the auto_align context manager."""
    from polder.config import auto_align

    # Default should be True.
    assert auto_align() is True

    # Inside context manager, should be False.
    with auto_align(False):
        assert auto_align() is False

    # After context manager, should be back to True.
    assert auto_align() is True


def test_auto_align_nested_contexts():
    """Test nested auto_align context managers."""
    from polder.config import auto_align

    assert auto_align() is True

    with auto_align(False):
        assert auto_align() is False

        with auto_align(True):
            assert auto_align() is True

        # Should be back to False after inner context.
        assert auto_align() is False

    # Should be back to True after outer context.
    assert auto_align() is True


def test_auto_align_with_already_aligned(value_array_type):
    """Test that auto_align(False) works with already aligned arrays."""
    from polder.config import auto_align

    array1 = generate_random_array(value_array_type, shape=(3, 4))
    array2 = generate_random_array(value_array_type, shape=(3, 4), seed=456)

    # With auto_align=False, adding already-aligned arrays should work.
    with auto_align(False):
        _ = array1 + array2


def test_auto_align_with_misaligned_raises(value_array_type):
    """Test that auto_align(False) raises with misaligned arrays."""
    from polder.config import auto_align

    array = generate_random_array(value_array_type, shape=(3, 4))
    misaligned = shuffle_labels(array)

    # With auto_align=True, the arrays can be added.
    _ = array + misaligned

    # With auto_align=False, adding misaligned arrays should raise.
    with auto_align(False):
        with pytest.raises(
            Exception, match="Cannot combine arrays with unaligned labels."
        ):
            _ = array + misaligned

import pytest

from .utils import ValueArrayType


@pytest.fixture(params=[ValueArrayType.NUMPY, ValueArrayType.JAX], scope="package")
def value_array_type(request):
    """Fixture that is parametrized on all array types."""
    return request.param

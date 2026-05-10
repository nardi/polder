from typing import TYPE_CHECKING, Any, Generic, Protocol, TypeAlias, TypeVar

from array_api_compat import array_namespace
from typing_extensions import TypeAliasType

# Only import the array API types if we are type checking, because the library
# uses Python 3.12 syntax. Otherwise we define dummy protocols that allow the
# same runtime usage.
if TYPE_CHECKING:
    from array_api._2024_12 import Array, ArrayNamespace
else:
    TDtype = TypeVar("TDtype")
    TDevice = TypeVar("TDevice")

    class Array(Generic[TDtype, TDevice], Protocol):
        pass

    TArray = TypeVar("TArray", bound=Array)

    class ArrayNamespace(Generic[TArray, TDtype, TDevice], Protocol):
        pass


AnyValueArray: TypeAlias = Array[Any, Any]

SomeValueArray = TypeVar("SomeValueArray", bound=AnyValueArray)

ValueArrayNamespace = TypeAliasType(
    "ValueArrayNamespace",
    ArrayNamespace[SomeValueArray, Any, Any],
    type_params=(SomeValueArray,),
)

__all__ = [
    "Array",
    "ArrayNamespace",
    "AnyValueArray",
    "SomeValueArray",
    "ValueArrayNamespace",
    "array_equal",
]


def array_equal(
    a1: SomeValueArray, a2: SomeValueArray, equal_nan: bool = False
) -> bool:
    """Reproduction of numpy.array_equal using the array API interface."""
    xp = array_namespace(a1, a2)

    if a1.shape != a2.shape:
        return False

    if not equal_nan:
        return bool(xp.all(a1 == a2))

    if a1 is a2:
        # nan will compare equal so an array will compare equal to itself.
        return True

    # Handling NaN values if equal_nan is True
    a1nan, a2nan = xp.isnan(a1), xp.isnan(a2)

    # NaN's occur at different locations
    if not xp.all(a1nan == a2nan):
        return False

    nan_mask = a1nan  # or a2nan

    # Indexing the non-NaN values of a1/a2 gives a dynamic shape. Instead we
    # create regular equality result, and fill in true for the places where we
    # know both are NaN.
    elements_eq = a1 == a2
    elements_eq = xp.where(nan_mask, True, elements_eq)

    return bool(xp.all(elements_eq))

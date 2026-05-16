from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Generic, Protocol, Self, TypeAlias, TypeVar, overload

from narwhals import DataFrame, Expr
from typing_extensions import TypeAliasType

from polder.protocols.descriptor import Descriptor

AnyDataFrame: TypeAlias = DataFrame[Any]


class AnyArray(Protocol):
    """A protocol that value array types have to satisfy.."""


LabelFrameType = TypeVar("LabelFrameType", bound=AnyDataFrame, covariant=True)
ValueArrayType = TypeVar("ValueArrayType", bound=AnyArray)


ArrayAxisIndices = TypeAliasType(
    "ArrayAxisIndices", int | slice | ValueArrayType, type_params=(ValueArrayType,)
)
LabelAxisIndices: TypeAlias = Mapping[str, Any] | Expr
AxisIndices = TypeAliasType(
    "AxisIndices",
    ArrayAxisIndices[ValueArrayType] | LabelAxisIndices,
    type_params=(ValueArrayType,),
)

Scalar: TypeAlias = int | float | complex | bool

AxisLabelsSpecifier: TypeAlias = Sequence[str]
AxisLabelsToPivot: TypeAlias = AxisLabelsSpecifier | Sequence[AxisLabelsSpecifier]
AxesSlice = tuple[int, int]


class FrameLabeledArray(Generic[LabelFrameType, ValueArrayType], Protocol):
    values: Descriptor[Self, ValuesIndexer[ValueArrayType]]
    """The value array that is being labeled. Can be any type that supports
    the Array API. Allows Numpy-style indexing to return a part of the
    array, which may be more efficient."""

    @overload
    def labels(self, axis: int) -> LabelFrameType | None: ...

    @overload
    def labels(self, axis: slice | None = ...) -> Sequence[LabelFrameType | None]: ...

    def labels(
        self, axis: int | slice | None = None
    ) -> LabelFrameType | None | Sequence[LabelFrameType | None]:
        """The labels for the value array. There is one frame per axis of the
        array, so `len(labels) == len(values.shape)`. If an axis has size 1, it
        may be unlabeled, represented by `None`. Allows returning labels for
        only a part of the axes, which may be more efficient."""
        ...

    def shape(self) -> Sequence[int]:
        """The shape of the array."""
        ...

    def __getitem__(
        self,
        indices: AxisIndices[ValueArrayType] | tuple[AxisIndices[ValueArrayType], ...],
    ) -> Self:
        """Index the array, either using numerical indices, or by filtering the
        labels.

        Given an N-dimensional array, there are a few different methods of
        indexing:
          1. Single-valued numerical indexing: when an axis is indexed with a
             single number, the N-1 dimensional slice at that location is
             returned.
          2. Multi-valued numerical indexing: when an axis is indexed with a
             slice or a numerical array (of same type as `values`), the array is
             sliced along that axis and the N-dimensional result is returned.
          3. Simple label filtering: when an axis is indexed with a mapping
             (e.g. `dict`), it is used to filter the labels of that axis. The
             keys are taken to refer to columns in the label frame, and the
             values to the single value that is kept for that column. After
             this, the column is removed from the label frame (analogous to type
             1 indexing). If it is the only column in the label frame, the whole
             axis is removed and the result will be N-1 dimensional.
          4. Complex label filtering: when an axis is indexed with a Narwhals
             expression, that expression is used to `filter` the label frame for
             that axis. The width of the label frame or dimensionality of the
             array is not changed.
          5. Axis creation indexing: when `None` is used to index the array at
             position `i`, a new size-1 unlabeled axis will be created between
             the current axes `i-1` and `i`. This can be useful to perform
             broadcasting."""
        ...

    def equals(self, other: Self, /) -> bool: ...

    def pivot(
        self,
        /,
        *,
        axis_labels_to_pivot: Mapping[int, AxisLabelsToPivot],
        fill_value: Any = ...,
    ) -> Self: ...

    def unpivot(
        self,
        /,
        *,
        axes_to_merge: Sequence[AxesSlice],
    ) -> Self: ...

    # Arithmetic operators
    def __abs__(self, /) -> Self: ...
    def __pos__(self, /) -> Self: ...
    def __neg__(self, /) -> Self: ...
    def __add__(self, other: Self | Scalar, /) -> Self: ...
    def __radd__(self, other: Self | Scalar, /) -> Self: ...
    def __sub__(self, other: Self | Scalar, /) -> Self: ...
    def __rsub__(self, other: Self | Scalar, /) -> Self: ...
    def __mul__(self, other: Self | Scalar, /) -> Self: ...
    def __rmul__(self, other: Self | Scalar, /) -> Self: ...
    def __truediv__(self, other: Self | Scalar, /) -> Self: ...
    def __rtruediv__(self, other: Self | Scalar, /) -> Self: ...
    def __floordiv__(self, other: Self | Scalar, /) -> Self: ...
    def __rfloordiv__(self, other: Self | Scalar, /) -> Self: ...
    def __mod__(self, other: Self | Scalar, /) -> Self: ...
    def __rmod__(self, other: Self | Scalar, /) -> Self: ...
    def __pow__(self, other: Self | Scalar, /) -> Self: ...
    def __rpow__(self, other: Self | Scalar, /) -> Self: ...

    # Matrix multiplication
    def __matmul__(self, other: Self, /) -> Self: ...

    # Bitwise operators
    def __invert__(self, /) -> Self: ...
    def __and__(self, other: Self | Scalar, /) -> Self: ...
    def __rand__(self, other: Self | Scalar, /) -> Self: ...
    def __or__(self, other: Self | Scalar, /) -> Self: ...
    def __ror__(self, other: Self | Scalar, /) -> Self: ...
    def __xor__(self, other: Self | Scalar, /) -> Self: ...
    def __rxor__(self, other: Self | Scalar, /) -> Self: ...
    def __lshift__(self, other: Self | Scalar, /) -> Self: ...
    def __rlshift__(self, other: Self | Scalar, /) -> Self: ...
    def __rshift__(self, other: Self | Scalar, /) -> Self: ...
    def __rrshift__(self, other: Self | Scalar, /) -> Self: ...

    # Comparison operators
    def __lt__(self, other: Self | Scalar, /) -> Self: ...
    def __le__(self, other: Self | Scalar, /) -> Self: ...
    def __gt__(self, other: Self | Scalar, /) -> Self: ...
    def __ge__(self, other: Self | Scalar, /) -> Self: ...
    # Apparently __eq__ is supposed to return boolean, does Numpy break this rule too?
    def __eq__(self, other: Self | Scalar, /) -> Self: ...  # type: ignore
    def __ne__(self, other: Self | Scalar, /) -> Self: ...  # type: ignore


AnyFrameLabeledArray: TypeAlias = FrameLabeledArray[Any, Any]

SomeFrameLabeledArray = TypeVar("SomeFrameLabeledArray", bound=AnyFrameLabeledArray)


class ValuesIndexer(Generic[ValueArrayType], Protocol):
    def __call__(
        self, *indices: ArrayAxisIndices[ValueArrayType]
    ) -> ValueArrayType: ...

    def __getitem__(
        self,
        indices: ArrayAxisIndices[ValueArrayType]
        | tuple[ArrayAxisIndices[ValueArrayType], ...],
        /,
    ) -> ValueArrayType: ...

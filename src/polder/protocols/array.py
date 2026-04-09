from collections.abc import Sequence
from typing import Any, Generic, Protocol, Self, TypeAlias, TypeVar

from narwhals import DataFrame, Expr
from optype.numpy import Array1D, CanArray

AnyDataFrame: TypeAlias = DataFrame[Any]


class AnyArray(CanArray, Protocol):
    """A protocol that value array types have to satisfy.."""


LabelFrameType = TypeVar("LabelFrameType", bound=AnyDataFrame, covariant=True)
ValueArrayType = TypeVar("ValueArrayType", bound=AnyArray, covariant=True)


AxisIndices = int | slice | list[int] | Array1D | Expr


class FrameLabeledArray(Generic[LabelFrameType, ValueArrayType], Protocol):
    def values(self) -> ValueArrayType:
        """The value array that is being labeled. Can be any type that supports the Array API."""
        ...

    def labels(
        self, axis: int | slice | None = None
    ) -> LabelFrameType | None | Sequence[LabelFrameType | None]:
        """The labels for the value array. There is one frame per axis of the array, so `len(labels)
        == len(values.shape)`. If an axis has size 1, it may be unlabeled, represented by `None`."""
        ...

    def shape(self) -> Sequence[int]:
        """The shape of the array."""
        ...

    def __getitem__(self, indices: AxisIndices | tuple[AxisIndices, ...]) -> Self:
        """Index the array, either using numerical indices, or using Narwhals expressions which will
        be used to filter the labels."""
        ...


AnyFrameLabeledArray: TypeAlias = FrameLabeledArray[AnyDataFrame, AnyArray]

SomeFrameLabeledArray = TypeVar("SomeFrameLabeledArray", bound="AnyFrameLabeledArray")

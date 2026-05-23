from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, overload

import narwhals as nw

from polder.eager.value_array import NumpyArray, SomeValueArray

if TYPE_CHECKING:
    import polars as pl

from polder.eager.array import EagerFrameLabeledArray, SomeEagerFrameLabeledArray
from polder.lazy.array import (
    ExternalFrameType,
    LazyFrameLabeledArray,
    SomeLazyFrameLabeledArray,
)
from polder.protocols.array import FrameLabeledArray, LabelFrameType, ValueArrayType
from polder.protocols.implementations import (
    EAGER,
    FrameLabeledArrayImplementation,
)


@overload
def convert(
    array: SomeLazyFrameLabeledArray,
    *,
    implementation: Literal[FrameLabeledArrayImplementation.LAZY],
) -> SomeLazyFrameLabeledArray: ...


@overload
def convert(
    array: SomeEagerFrameLabeledArray,
    *,
    implementation: Literal[FrameLabeledArrayImplementation.EAGER],
) -> SomeEagerFrameLabeledArray: ...


@overload
def convert(
    array: LazyFrameLabeledArray[ExternalFrameType, Any],
    *,
    implementation: Literal[FrameLabeledArrayImplementation.EAGER],
) -> EagerFrameLabeledArray[ExternalFrameType, NumpyArray]: ...


if TYPE_CHECKING:

    @overload
    def convert(
        array: EagerFrameLabeledArray[nw.DataFrame[pl.DataFrame], Any],
        *,
        implementation: Literal[FrameLabeledArrayImplementation.LAZY],
    ) -> LazyFrameLabeledArray[
        nw.DataFrame[pl.DataFrame], nw.LazyFrame[pl.LazyFrame]
    ]: ...


@overload
def convert(
    array: EagerFrameLabeledArray[ExternalFrameType, Any],
    *,
    implementation: Literal[FrameLabeledArrayImplementation.LAZY],
) -> LazyFrameLabeledArray[ExternalFrameType, Any]: ...


@overload
def convert(
    array: FrameLabeledArray[LabelFrameType, ValueArrayType],
    *,
    implementation: Literal[FrameLabeledArrayImplementation.LAZY],
) -> LazyFrameLabeledArray[LabelFrameType, Any]: ...


@overload
def convert(
    array: FrameLabeledArray[LabelFrameType, SomeValueArray],
    *,
    implementation: Literal[FrameLabeledArrayImplementation.EAGER],
) -> EagerFrameLabeledArray[LabelFrameType, SomeValueArray]: ...


@overload
def convert(
    array: FrameLabeledArray[LabelFrameType, Any],
    *,
    implementation: Literal[FrameLabeledArrayImplementation.EAGER],
) -> EagerFrameLabeledArray[LabelFrameType, Any]: ...


def convert(
    array: FrameLabeledArray[Any, Any],
    *,
    implementation: FrameLabeledArrayImplementation = EAGER,
) -> FrameLabeledArray[Any, Any]:
    """Convert a FrameLabeledArray to a different implementation.

    If the array is already in the target implementation, it is returned
    unchanged without copying any data.

    Args:
        array: The array to convert.
        implementation: The target implementation. Defaults to EAGER.

    Returns:
        The array in the target implementation.
    """
    match implementation:
        case FrameLabeledArrayImplementation.EAGER:
            if isinstance(array, EagerFrameLabeledArray):
                return array
            return EagerFrameLabeledArray.from_values_and_labels(
                array.values(), array.labels()
            )
        case FrameLabeledArrayImplementation.LAZY:
            if isinstance(array, LazyFrameLabeledArray):
                return array
            return LazyFrameLabeledArray.from_values_and_labels(
                array.values(), array.labels()
            )
    raise NotImplementedError()

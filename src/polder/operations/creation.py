from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Literal, overload

import narwhals as nw
import numpy as np
from narwhals.typing import IntoDataFrameT

if TYPE_CHECKING:
    import polars as pl

from polder.eager.array import EagerFrameLabeledArray
from polder.eager.value_array import SomeValueArray
from polder.lazy.array import LazyFrameLabeledArray
from polder.protocols.array import AnyArray, FrameLabeledArray
from polder.protocols.implementations import (
    EAGER,
    LAZY,
    FrameLabeledArrayImplementation,
)


@overload
def from_values_and_labels(
    values: AnyArray,
    labels: Iterable[pl.DataFrame | None],
    *,
    implementation: Literal[FrameLabeledArrayImplementation.LAZY],
) -> LazyFrameLabeledArray[pl.DataFrame, pl.LazyFrame]: ...


@overload
def from_values_and_labels(
    values: AnyArray,
    labels: Iterable[IntoDataFrameT | None],
    *,
    implementation: Literal[FrameLabeledArrayImplementation.LAZY],
) -> LazyFrameLabeledArray[IntoDataFrameT, Any]: ...


@overload
def from_values_and_labels(
    values: SomeValueArray,
    labels: Iterable[IntoDataFrameT | None],
    *,
    implementation: Literal[FrameLabeledArrayImplementation.EAGER] = ...,
) -> EagerFrameLabeledArray[nw.DataFrame[IntoDataFrameT], SomeValueArray]: ...


# We add an overload for np.ndarray specifically because the concrete class
# doesn't validate against Array from `types-array-api`. This way we can do a
# typecast here and let both library and user code be typed correctly.
@overload
def from_values_and_labels(
    values: np.ndarray,
    labels: Iterable[IntoDataFrameT | None],
    *,
    implementation: Literal[FrameLabeledArrayImplementation.EAGER] = ...,
) -> FrameLabeledArray[nw.DataFrame[IntoDataFrameT], np.ndarray]: ...


# JAX is an optional dependency, so we guard the import with a try-catch block.
maybe_jax = None
try:
    import jax

    maybe_jax = jax

    @overload
    def from_values_and_labels(
        values: jax.Array,
        labels: Iterable[IntoDataFrameT | None],
        *,
        implementation: Literal[FrameLabeledArrayImplementation.EAGER] = ...,
    ) -> FrameLabeledArray[nw.DataFrame[IntoDataFrameT], jax.Array]: ...
except ImportError:
    pass


def from_values_and_labels(
    values: AnyArray,
    labels: Iterable[IntoDataFrameT | None],
    *,
    implementation: FrameLabeledArrayImplementation = EAGER,
) -> FrameLabeledArray[nw.DataFrame[IntoDataFrameT], Any]:
    label_dfs = tuple(
        nw.from_native(axis_labels) if axis_labels is not None else None
        for axis_labels in labels
    )
    match implementation:
        case FrameLabeledArrayImplementation.EAGER:
            if isinstance(values, np.ndarray):
                # np.ndarray doesn't type as Array for some reason.
                return EagerFrameLabeledArray.from_values_and_labels(values, label_dfs)  # type: ignore
            if maybe_jax is not None and isinstance(values, maybe_jax.Array):
                # jax.Array also doesn't type as Array.
                return EagerFrameLabeledArray.from_values_and_labels(values, label_dfs)  # type: ignore
        case FrameLabeledArrayImplementation.LAZY:
            return LazyFrameLabeledArray.from_values_and_labels(values, label_dfs)
    raise NotImplementedError()


@overload
def from_frame(
    frame: nw.DataFrame[pl.DataFrame],
    *,
    value_column: str = "value",
    implementation: Literal[FrameLabeledArrayImplementation.LAZY] = ...,
) -> LazyFrameLabeledArray[pl.DataFrame, pl.LazyFrame]: ...


@overload
def from_frame(
    frame: nw.DataFrame[IntoDataFrameT],
    *,
    value_column: str = "value",
    implementation: Literal[FrameLabeledArrayImplementation.LAZY] = ...,
) -> LazyFrameLabeledArray[IntoDataFrameT, Any]: ...


def from_frame(
    frame: nw.DataFrame[IntoDataFrameT],
    *,
    value_column: str = "value",
    implementation: FrameLabeledArrayImplementation = LAZY,
) -> FrameLabeledArray[nw.DataFrame[IntoDataFrameT], np.ndarray]:
    lazy_array = LazyFrameLabeledArray.from_frame(frame, value_column=value_column)

    match implementation:
        case FrameLabeledArrayImplementation.LAZY:
            return lazy_array
        case FrameLabeledArrayImplementation.EAGER:
            raise NotImplementedError()

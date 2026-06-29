from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Literal, overload

import narwhals as nw
import numpy as np
from narwhals.typing import IntoDataFrameT

if TYPE_CHECKING:
    import polars as pl

from polder.eager.array import EagerFrameLabeledArray
from polder.eager.value_array import JaxArray, NumpyArray, SomeValueArray
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
) -> LazyFrameLabeledArray[nw.DataFrame[pl.DataFrame], nw.LazyFrame[pl.LazyFrame]]: ...


@overload
def from_values_and_labels(
    values: AnyArray,
    labels: Iterable[IntoDataFrameT | None],
    *,
    implementation: Literal[FrameLabeledArrayImplementation.LAZY],
) -> LazyFrameLabeledArray[nw.DataFrame[IntoDataFrameT], Any]: ...


@overload
def from_values_and_labels(
    values: SomeValueArray,
    labels: Iterable[IntoDataFrameT | None],
    *,
    implementation: Literal[FrameLabeledArrayImplementation.EAGER] = ...,
) -> EagerFrameLabeledArray[nw.DataFrame[IntoDataFrameT], SomeValueArray]: ...


@overload
def from_values_and_labels(
    values: np.ndarray,
    labels: Iterable[IntoDataFrameT | None],
    *,
    implementation: Literal[FrameLabeledArrayImplementation.EAGER] = ...,
) -> FrameLabeledArray[nw.DataFrame[IntoDataFrameT], NumpyArray]: ...


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
    ) -> FrameLabeledArray[nw.DataFrame[IntoDataFrameT], JaxArray]: ...

except ImportError:
    pass


def from_values_and_labels(
    values: AnyArray,
    labels: Iterable[IntoDataFrameT | None],
    *,
    implementation: FrameLabeledArrayImplementation = EAGER,
) -> FrameLabeledArray[nw.DataFrame[IntoDataFrameT], Any]:
    """Create a frame-labeled array from a value array and a label frame per axis.

    There is one label frame per axis, so the number of frames must equal the number of
    dimensions of `values`, and the number of rows of each frame must match the size of
    the corresponding axis. A frame may have more than one column, which attaches several
    labels to the same axis. An axis of size 1 may be left unlabeled by passing None for
    its frame, which marks it as broadcastable.

    Args:
        values: The values to label. Any array following the array API is accepted by the
            eager implementation. The lazy implementation converts the values to NumPy.
        labels: One label frame (or None) per axis, in axis order. Each frame is anything
            Narwhals can wrap, such as a Polars DataFrame.
        implementation: The implementation to build. Defaults to EAGER.

    Returns:
        A frame-labeled array of the requested implementation, wrapping the given values
        and labels.
    """
    label_dfs = tuple(
        nw.from_native(axis_labels) if axis_labels is not None else None
        for axis_labels in labels
    )
    match implementation:
        case FrameLabeledArrayImplementation.EAGER:
            if isinstance(values, (NumpyArray, JaxArray)):
                return EagerFrameLabeledArray.from_values_and_labels(values, label_dfs)
        case FrameLabeledArrayImplementation.LAZY:
            return LazyFrameLabeledArray.from_values_and_labels(values, label_dfs)
    raise NotImplementedError()


@overload
def from_frame(
    frame: nw.DataFrame[pl.DataFrame],
    *,
    value_column: str = "value",
    implementation: Literal[FrameLabeledArrayImplementation.LAZY] = ...,
) -> LazyFrameLabeledArray[nw.DataFrame[pl.DataFrame], nw.LazyFrame[pl.LazyFrame]]: ...


@overload
def from_frame(
    frame: nw.DataFrame[IntoDataFrameT],
    *,
    value_column: str = "value",
    implementation: Literal[FrameLabeledArrayImplementation.LAZY] = ...,
) -> LazyFrameLabeledArray[nw.DataFrame[IntoDataFrameT], Any]: ...


def from_frame(
    frame: nw.DataFrame[IntoDataFrameT],
    *,
    value_column: str = "value",
    implementation: FrameLabeledArrayImplementation = LAZY,
) -> FrameLabeledArray[nw.DataFrame[IntoDataFrameT], np.ndarray]:
    """Create a one-dimensional frame-labeled array from a single frame in long format.

    One column of the frame holds the values, and the remaining columns become the labels
    of the single axis. The resulting array is always one-dimensional. To obtain a
    higher-dimensional array, use `from_values_and_labels` or reshape with `pivot`.

    Args:
        frame: A Narwhals DataFrame in long format, with one row per array element.
        value_column: The name of the column holding the values. Defaults to "value".
        implementation: The implementation to build. Defaults to LAZY, since keeping the
            data in its DataFrame backend is the typical reason to start from a frame. The
            EAGER implementation is not yet supported by this function.

    Returns:
        A one-dimensional frame-labeled array of the requested implementation.

    Raises:
        NotImplementedError: If the EAGER implementation is requested.
    """
    lazy_array = LazyFrameLabeledArray.from_frame(frame, value_column=value_column)

    match implementation:
        case FrameLabeledArrayImplementation.LAZY:
            return lazy_array
        case FrameLabeledArrayImplementation.EAGER:
            raise NotImplementedError()

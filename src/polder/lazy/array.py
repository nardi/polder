from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property
from types import ModuleType
from typing import Any, Generic, Self, TypeAlias, TypeVar, cast, overload

import narwhals as nw
import numpy as np

from polder.config import use_eager_evaluation_for_lazy_arrays
from polder.protocols.array import AnyArray, ArrayAxisIndices, FrameLabeledArray
from polder.utils.indexer import indexermethod

AnyExternalFrame: TypeAlias = nw.DataFrame[Any]
ExternalFrameType = TypeVar("ExternalFrameType", bound=AnyExternalFrame)
AnyInternalFrame: TypeAlias = nw.LazyFrame[Any] | nw.DataFrame[Any]
InternalFrameType = TypeVar("InternalFrameType", bound=AnyInternalFrame)


def _create_index_df(shape: Sequence[int], frame_ns: ModuleType):
    n_dims = len(shape)
    indices = np.indices(shape).reshape((n_dims), -1).T
    return nw.from_numpy(
        indices,
        schema=[f"__index{i}" for i in range(n_dims)],
        backend=frame_ns,
    )


@dataclass(frozen=True, eq=False, slots=True)
class LazyFrameLabeledArray(
    Generic[ExternalFrameType, InternalFrameType],
    FrameLabeledArray[ExternalFrameType, np.ndarray],
):
    """A variant of FrameLabeledArray that stores all data (labels and values)
    in a table (long) format, and resolves into an array lazily."""

    _indexed_labels: tuple[InternalFrameType | None, ...]
    """The labels for each axis in the array. These contain an additional column
    `__index` which specifies the ordering."""
    _values: InternalFrameType
    """The values for the array. These are stored in long format, with an
    index-column for each axis named `__index0`, `__index1`, etc. and a single
    `value` column."""
    _shape: InternalFrameType
    """The shape of the array. Because the values are lazily computed, this is a
    separate computation that guarantees the validity of the array. This is a
    table with two columns, `axis` and `size`."""
    _n_dims: int
    """The number of dimensions of the array."""
    _frame_ns: ModuleType
    """The native namespace used to construct Narwhals DataFrames."""

    @staticmethod
    def from_values_and_labels(
        values: AnyArray, labels: Sequence[ExternalFrameType | None]
    ) -> LazyFrameLabeledArray[ExternalFrameType, AnyInternalFrame]:
        frame_ns = next(
            (df.implementation for df in labels if df is not None),
            nw.Implementation.POLARS,
        ).to_native_namespace()

        values = np.asarray(values)

        n_dims = len(values.shape)

        use_lazy_frames = not use_eager_evaluation_for_lazy_arrays()

        def maybe_lazy(frame: AnyExternalFrame) -> AnyInternalFrame:
            if use_lazy_frames:
                return frame.lazy()
            return frame

        shape_frame = maybe_lazy(
            nw.from_numpy(
                np.stack([np.arange(n_dims), np.array(values.shape)], axis=1),
                schema=["axis", "size"],
                backend=frame_ns,
            )
        )

        values_frame = maybe_lazy(
            _create_index_df(values.shape, frame_ns).with_columns(
                value=values.reshape((-1,))
            )
        )

        indexed_labels = tuple(
            maybe_lazy(axis_labels.with_row_index("__index"))
            if axis_labels is not None
            else None
            for axis_labels in labels
        )

        return LazyFrameLabeledArray(
            indexed_labels, values_frame, shape_frame, n_dims, frame_ns
        )

    @staticmethod
    def from_frame(
        frame: ExternalFrameType, *, value_column: str = "value"
    ) -> LazyFrameLabeledArray[ExternalFrameType, AnyInternalFrame]:
        frame_ns = frame.implementation.to_native_namespace()

        use_lazy_frames = not use_eager_evaluation_for_lazy_arrays()

        def maybe_lazy(frame: AnyExternalFrame) -> AnyInternalFrame:
            if use_lazy_frames:
                return frame.lazy()
            return frame

        # An array created from a frame will always be 1-dimensional.
        n_dims = 1
        shape = maybe_lazy(
            nw.from_dict({"axis": [0], "size": [len(frame)]}, backend=frame_ns)
        )

        # Split the frame into labels and values, addding an index to each.
        labels = (
            maybe_lazy(
                frame.select(nw.exclude(value_column)).with_row_index("__index")
            ),
        )
        values = maybe_lazy(
            frame
            .select(value_column)
            .rename({value_column: "value"})
            .with_row_index("__index0")
        )

        return LazyFrameLabeledArray(labels, values, shape, n_dims, frame_ns)

    @indexermethod
    def values(self, *indices: ArrayAxisIndices) -> np.ndarray:
        value_index = _create_index_df(self.shape(), self._frame_ns)
        index_columns = value_index.columns
        indexed_values = (
            value_index
            .lazy()
            .join(self._values.lazy(), on=index_columns, how="left")
            .sort(index_columns)
            .collect()
        )

        if indexed_values.select(
            nw.any_horizontal(nw.col("value").is_null().any(), ignore_nulls=True)
        ).item():
            raise ValueError("Array has indices for which no value is stored")

        values = indexed_values["value"].to_numpy().reshape(self.shape())

        # TODO: filter on indices before converting to array.
        values = values[indices or ...]

        return values

    @overload
    def labels(self, axis: int) -> ExternalFrameType | None: ...

    @overload
    def labels(
        self, axis: slice | None = ...
    ) -> Sequence[ExternalFrameType | None]: ...

    def labels(
        self, axis: int | slice | None = None
    ) -> ExternalFrameType | None | Sequence[ExternalFrameType | None]:
        if axis is None:
            axis = slice(None)

        selected_labels = self._indexed_labels[axis]

        def collect_labels(
            axis_labels: AnyInternalFrame,
        ) -> ExternalFrameType:
            # We ensure we have the right DataFrame type, but this cannot be
            # deduced by the type checker, so we do a cast.
            return cast(
                ExternalFrameType,
                axis_labels
                .lazy()
                .sort("__index")
                .drop("__index")
                .collect(backend=self._frame_ns),
            )

        if isinstance(selected_labels, tuple):
            return tuple(
                collect_labels(axis_labels) if axis_labels is not None else None
                for axis_labels in selected_labels
            )
        elif isinstance(selected_labels, (nw.DataFrame, nw.LazyFrame)):
            return collect_labels(selected_labels)

        return None

    def shape(self) -> Sequence[int]:
        return self._evaluated_shape

    @cached_property
    def _evaluated_shape(self) -> Sequence[int]:
        axes = nw.from_numpy(
            np.arange(self._n_dims, dtype=int)[:, None],
            schema=["axis"],
            backend=self._frame_ns,
        )
        shape = (
            axes
            .lazy()
            .join(self._shape.lazy(), on="axis", how="left")
            .sort("axis")
            .collect()
        )

        is_valid_shape = (
            len(shape) == len(axes)
            and not shape.select(
                nw.any_horizontal(nw.col("size").is_null().any(), ignore_nulls=True)
            ).item()
        )

        if not is_valid_shape:
            raise ValueError("Array has invalid shape")

        return tuple(shape["size"])

    def __getitem__(
        self,
        indices: int
        | slice[Any, Any, Any]
        | Any
        | nw.Expr
        | tuple[int | slice[Any, Any, Any] | Any | nw.Expr, ...],
    ) -> Self:
        raise NotImplementedError

    def equals(self, other: Self) -> bool:
        raise NotImplementedError

    def pivot(
        self,
        /,
        *,
        axis_labels_to_pivot: Mapping[int, Sequence[str] | Sequence[Sequence[str]]],
        fill_value: Any = ...,
    ) -> Self:
        raise NotImplementedError

    def unpivot(self, /, *, axes_to_merge: Sequence[tuple[int, int]]) -> Self:
        raise NotImplementedError

    def __abs__(self) -> Self:
        raise NotImplementedError

    def __pos__(self) -> Self:
        raise NotImplementedError

    def __neg__(self) -> Self:
        raise NotImplementedError

    def __add__(self, other: Self | int | float | complex | bool) -> Self:
        raise NotImplementedError

    def __radd__(self, other: Self | int | float | complex | bool) -> Self:
        raise NotImplementedError

    def __sub__(self, other: Self | int | float | complex | bool) -> Self:
        raise NotImplementedError

    def __rsub__(self, other: Self | int | float | complex | bool) -> Self:
        raise NotImplementedError

    def __mul__(self, other: Self | int | float | complex | bool) -> Self:
        raise NotImplementedError

    def __rmul__(self, other: Self | int | float | complex | bool) -> Self:
        raise NotImplementedError

    def __truediv__(self, other: Self | int | float | complex | bool) -> Self:
        raise NotImplementedError

    def __rtruediv__(self, other: Self | int | float | complex | bool) -> Self:
        raise NotImplementedError

    def __floordiv__(self, other: Self | int | float | complex | bool) -> Self:
        raise NotImplementedError

    def __rfloordiv__(self, other: Self | int | float | complex | bool) -> Self:
        raise NotImplementedError

    def __mod__(self, other: Self | int | float | complex | bool) -> Self:
        raise NotImplementedError

    def __rmod__(self, other: Self | int | float | complex | bool) -> Self:
        raise NotImplementedError

    def __pow__(self, other: Self | int | float | complex | bool) -> Self:
        raise NotImplementedError

    def __rpow__(self, other: Self | int | float | complex | bool) -> Self:
        raise NotImplementedError

    def __matmul__(self, other: Self) -> Self:
        raise NotImplementedError

    def __invert__(self) -> Self:
        raise NotImplementedError

    def __and__(self, other: Self | int | float | complex | bool) -> Self:
        raise NotImplementedError

    def __rand__(self, other: Self | int | float | complex | bool) -> Self:
        raise NotImplementedError

    def __or__(self, other: Self | int | float | complex | bool) -> Self:
        raise NotImplementedError

    def __ror__(self, other: Self | int | float | complex | bool) -> Self:
        raise NotImplementedError

    def __xor__(self, other: Self | int | float | complex | bool) -> Self:
        raise NotImplementedError

    def __rxor__(self, other: Self | int | float | complex | bool) -> Self:
        raise NotImplementedError

    def __lshift__(self, other: Self | int | float | complex | bool) -> Self:
        raise NotImplementedError

    def __rlshift__(self, other: Self | int | float | complex | bool) -> Self:
        raise NotImplementedError

    def __rshift__(self, other: Self | int | float | complex | bool) -> Self:
        raise NotImplementedError

    def __rrshift__(self, other: Self | int | float | complex | bool) -> Self:
        raise NotImplementedError

    def __lt__(self, other: Self | int | float | complex | bool) -> Self:
        raise NotImplementedError

    def __le__(self, other: Self | int | float | complex | bool) -> Self:
        raise NotImplementedError

    def __gt__(self, other: Self | int | float | complex | bool) -> Self:
        raise NotImplementedError

    def __ge__(self, other: Self | int | float | complex | bool) -> Self:
        raise NotImplementedError

    def __eq__(self, other: Self | int | float | complex | bool) -> Self:
        raise NotImplementedError

    def __ne__(self, other: Self | int | float | complex | bool) -> Self:
        raise NotImplementedError


AnyLazyFrameLabeledArray = LazyFrameLabeledArray[AnyExternalFrame, AnyInternalFrame]
SomeLazyFrameLabeledArray = TypeVar(
    "SomeLazyFrameLabeledArray", bound=AnyLazyFrameLabeledArray
)

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import ModuleType
from typing import Any, Generic, Self, TypeAlias, TypeVar, cast, overload

import narwhals as nw
import narwhals.typing as nwt
import numpy as np

from polder.protocols.array import AnyArray, ArrayAxisIndices, FrameLabeledArray

AnyExternalFrame: TypeAlias = nwt.IntoDataFrame
ExternalFrameType = TypeVar("ExternalFrameType", bound=AnyExternalFrame)
AnyInternalFrame: TypeAlias = nwt.IntoLazyFrame
InternalFrameType = TypeVar("InternalFrameType", bound=AnyInternalFrame)


@dataclass(frozen=True, eq=False, slots=True)
class LazyFrameLabeledArray(
    Generic[ExternalFrameType, InternalFrameType],
    FrameLabeledArray[nw.DataFrame[ExternalFrameType], np.ndarray],
):
    """A variant of FrameLabeledArray that stores all data (labels and values)
    in a table (long) format, and resolves into an array lazily."""

    _indexed_labels: tuple[nw.LazyFrame[InternalFrameType] | None, ...]
    """The labels for each axis in the array. These contain an additional column
    `__index` which specifies the ordering."""
    _values: nw.LazyFrame[InternalFrameType]
    """The values for the array. These are stored in long format, with an
    index-column for each axis named `__index0`, `__index1`, etc. and a single
    `value` column."""
    _shape: nw.LazyFrame[InternalFrameType]
    """The shape of the array. Because the values are lazily computed, this is a
    separate computation that guarantees the validity of the array. This is a
    table with two columns, `axis` and `size`."""
    _n_dims: int
    """The number of dimensions of the array."""
    _frame_ns: ModuleType
    """The native namespace used to construct Narwhals DataFrames."""

    @staticmethod
    def from_values_and_labels(
        values: AnyArray, labels: Sequence[nw.DataFrame[nwt.IntoDataFrameT] | None]
    ) -> LazyFrameLabeledArray[nwt.IntoDataFrameT, AnyInternalFrame]:
        frame_ns = next(
            (df.implementation for df in labels if df is not None),
            nw.Implementation.POLARS,
        ).to_native_namespace()

        values = np.asarray(values)

        n_dims = len(values.shape)

        shape_frame = nw.from_numpy(
            np.stack([np.arange(n_dims), np.array(values.shape)], axis=1),
            schema=["axis", "size"],
            backend=frame_ns,
        ).lazy()

        indexed_values = (
            np.concatenate([np.indices(values.shape), values[None]], axis=0)
            .reshape((n_dims + 1), -1)
            .T
        )
        values_frame = nw.from_numpy(
            indexed_values,
            schema=[*[f"__index{i}" for i in range(n_dims)], "value"],
            backend=frame_ns,
        ).lazy()

        indexed_labels = tuple(
            axis_labels.with_row_index("__index").lazy()
            if axis_labels is not None
            else None
            for axis_labels in labels
        )

        return LazyFrameLabeledArray(
            indexed_labels, values_frame, shape_frame, n_dims, frame_ns
        )

    @staticmethod
    def from_frame(
        frame: nw.DataFrame[nwt.IntoDataFrameT], *, value_column: str = "value"
    ) -> LazyFrameLabeledArray[nwt.IntoDataFrameT, AnyInternalFrame]:
        frame_ns = frame.implementation.to_native_namespace()

        # An array created from a frame will always be 1-dimensional.
        n_dims = 1
        shape = nw.from_dict(
            {"axis": [0], "size": [len(frame)]}, backend=frame_ns
        ).lazy()

        # Split the frame into labels and values, addding an index to each.
        labels = (
            frame.select(nw.exclude(value_column)).with_row_index("__index").lazy(),
        )
        values = (
            frame.select(value_column)
            .rename({value_column: "value"})
            .with_row_index("__index0")
            .lazy()
        )

        return LazyFrameLabeledArray(labels, values, shape, n_dims, frame_ns)

    def values(self, *indices: ArrayAxisIndices) -> np.ndarray:
        index_columns = [f"__index{i}" for i in range(self._n_dims)]
        values = (
            self._values.sort(index_columns)
            .drop(index_columns)
            .collect()["value"]
            .to_numpy()
            .reshape(self.shape())
        )

        # TODO: filter on indices before converting to array.
        values = values[indices or ...]

        return values

    @overload
    def labels(self, axis: int) -> nw.DataFrame[ExternalFrameType] | None: ...

    @overload
    def labels(
        self, axis: slice | None = ...
    ) -> Sequence[nw.DataFrame[ExternalFrameType] | None]: ...

    def labels(
        self, axis: int | slice | None = None
    ) -> (
        nw.DataFrame[ExternalFrameType]
        | None
        | Sequence[nw.DataFrame[ExternalFrameType] | None]
    ):
        if axis is None:
            axis = slice(None)

        selected_labels = self._indexed_labels[axis]

        def collect_labels(
            axis_labels: nw.LazyFrame,
        ) -> nw.DataFrame[ExternalFrameType]:
            # We ensure we have the right DataFrame type, but this cannot be
            # deduced by the type checker, so we do a cast.
            return cast(
                nw.DataFrame[ExternalFrameType],
                axis_labels.sort("__index")
                .drop("__index")
                .collect(backend=self._frame_ns),
            )

        if isinstance(selected_labels, tuple):
            return tuple(
                collect_labels(axis_labels) if axis_labels is not None else None
                for axis_labels in selected_labels
            )
        elif isinstance(selected_labels, nw.LazyFrame):
            return collect_labels(selected_labels)

        return None

    def shape(self) -> Sequence[int]:
        return tuple(self._shape.sort("axis").select("size").collect()["size"])

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

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property
from types import ModuleType
from typing import Any, Generic, Self, TypeAlias, TypeVar, cast, overload

import narwhals as nw
import numpy as np

import polder.lazy.binary as binary
import polder.lazy.unary as unary
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


def swap_args(f):
    return lambda a, b: f(b, a)


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
    def maybe_lazy(frame: AnyExternalFrame) -> AnyInternalFrame:
        use_lazy_frames = not use_eager_evaluation_for_lazy_arrays()

        if use_lazy_frames:
            return frame.lazy()
        return frame

    @classmethod
    def from_values_and_labels(
        cls, values: AnyArray, labels: Sequence[ExternalFrameType | None]
    ) -> LazyFrameLabeledArray[ExternalFrameType, AnyInternalFrame]:
        frame_ns = next(
            (df.implementation for df in labels if df is not None),
            nw.Implementation.POLARS,
        ).to_native_namespace()

        values = np.asarray(values)

        if np.iscomplexobj(values):
            raise NotImplementedError(
                "Lazy arrays do not support complex-valued arrays."
            )

        n_dims = len(values.shape)

        shape_frame = cls.maybe_lazy(
            nw.from_numpy(
                np.stack([np.arange(n_dims), np.array(values.shape)], axis=1),
                schema=["axis", "size"],
                backend=frame_ns,
            )
        )

        values_frame = cls.maybe_lazy(
            _create_index_df(values.shape, frame_ns).with_columns(
                value=values.reshape((-1,))
            )
        )

        indexed_labels = tuple(
            cls.maybe_lazy(axis_labels.with_row_index("__index"))
            if axis_labels is not None
            else None
            for axis_labels in labels
        )

        return LazyFrameLabeledArray(
            indexed_labels, values_frame, shape_frame, n_dims, frame_ns
        )

    @classmethod
    def from_frame(
        cls, frame: ExternalFrameType, *, value_column: str = "value"
    ) -> LazyFrameLabeledArray[ExternalFrameType, AnyInternalFrame]:
        frame_ns = frame.implementation.to_native_namespace()

        # An array created from a frame will always be 1-dimensional.
        n_dims = 1
        shape = cls.maybe_lazy(
            nw.from_dict({"axis": [0], "size": [len(frame)]}, backend=frame_ns)
        )

        # Split the frame into labels and values, addding an index to each.
        labels = (
            cls.maybe_lazy(
                frame.select(nw.exclude(value_column)).with_row_index("__index")
            ),
        )
        values = cls.maybe_lazy(
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

    def collect(self) -> Self:
        """Collect all LazyFrames backing this lazy array, resolving any lazy
        computations, and store the results in a new lazy array."""

        collect_frame = lambda frame: cast(
            InternalFrameType, self.maybe_lazy(frame.lazy().collect())
        )

        return type(self)(
            tuple(map(collect_frame, self._indexed_labels)),
            collect_frame(self._values),
            collect_frame(self._shape),
            self._n_dims,
            self._frame_ns,
        )

    # Arithmetic operators
    __abs__ = unary.abs_
    __pos__ = unary.pos
    __neg__ = unary.neg
    __add__ = binary.add
    __radd__ = binary.add
    __sub__ = binary.sub
    __rsub__ = swap_args(binary.sub)
    __mul__ = binary.mul
    __rmul__ = binary.mul
    __truediv__ = binary.truediv
    __rtruediv__ = swap_args(binary.truediv)
    __floordiv__ = binary.floordiv
    __rfloordiv__ = swap_args(binary.floordiv)
    __mod__ = binary.mod
    __rmod__ = swap_args(binary.mod)
    __pow__ = binary.pow
    __rpow__ = swap_args(binary.pow)

    # Bitwise operators
    __invert__ = unary.invert
    __and__ = binary.and_
    __rand__ = binary.and_
    __or__ = binary.or_
    __ror__ = binary.or_
    __xor__ = binary.xor
    __rxor__ = binary.xor
    __lshift__ = binary.lshift
    __rlshift__ = swap_args(binary.lshift)
    __rshift__ = binary.rshift
    __rrshift__ = swap_args(binary.rshift)

    # Comparison operators
    __lt__ = binary.lt
    __le__ = binary.le
    __gt__ = binary.gt
    __ge__ = binary.ge
    # Apparently __eq__ is supposed to return boolean, does Numpy break this rule too?
    __eq__ = binary.eq  # type: ignore
    __ne__ = binary.ne  # type: ignore

    ## Unimplemented protocol members ##

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

    def __matmul__(self, other: Self) -> Self:
        raise NotImplementedError


AnyLazyFrameLabeledArray = LazyFrameLabeledArray[Any, Any]
SomeLazyFrameLabeledArray = TypeVar(
    "SomeLazyFrameLabeledArray", bound=AnyLazyFrameLabeledArray
)

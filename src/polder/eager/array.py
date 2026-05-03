from dataclasses import dataclass
from typing import Self, TypeVar, overload

import numpy as np
from narwhals import Expr
from optype.numpy import Array

import polder.eager.binary as binary
import polder.eager.unary as unary
from polder.eager.pivot import pivot, unpivot
from polder.protocols.array import (
    AnyDataFrame,
    AxisIndices,
    FrameLabeledArray,
    LabelFrameType,
)


def swap_args(f):
    return lambda a, b: f(b, a)


@dataclass(frozen=True, eq=False)
class EagerFrameLabeledArray(FrameLabeledArray[LabelFrameType, Array]):
    _labels: tuple[LabelFrameType | None, ...]
    _values: Array

    def __post_init__(self):
        # Check that the labels and values have the same shape.
        labels_shape = tuple(
            len(labels) if labels is not None else 1 for labels in self._labels
        )
        assert labels_shape == self._values.shape

    @overload
    def labels(self, axis: int) -> LabelFrameType | None: ...

    @overload
    def labels(
        self, axis: slice | None = None
    ) -> tuple[LabelFrameType | None, ...]: ...

    def labels(
        self, axis: int | slice | None = None
    ) -> LabelFrameType | None | tuple[LabelFrameType | None, ...]:
        if axis is None:
            axis = slice(None)
        return self._labels[axis]

    def values(self) -> np.ndarray:
        return self._values

    def shape(self) -> tuple[int, ...]:
        return self._values.shape

    def __getitem__(self, indices: AxisIndices | tuple[AxisIndices, ...]) -> Self:
        if not isinstance(indices, tuple):
            indices = (indices,)
        assert len(indices) <= len(self._labels)

        labels = list(self._labels)
        values = self._values

        # Index each axis one-by-one.
        for i, idx in enumerate(indices):
            axis_labels = labels[i]

            # If the index is None, we want to add an unlabeled axis.
            if idx is None:
                labels.insert(i, None)
                value_idx = None

            # If we are indexing with an Expr, filter the axis labels and then use the result to
            # index the values.
            elif isinstance(idx, Expr):
                if axis_labels is None:
                    raise Exception("Cannot index unlabeled index with an expression")
                filtered_axis_labels = axis_labels.with_row_index(
                    "__value_index"
                ).filter(idx)
                labels[i] = filtered_axis_labels.select(axis_labels.columns)
                value_idx = filtered_axis_labels["__value_index"].to_numpy()

            # If we have a numerical indexer (int, int array or slice) we index both labels and
            # values with the indexer.
            else:
                if axis_labels is None and values.shape[i] != 1:
                    raise Exception("Cannot grow unlabeled index to multiple values")
                elif axis_labels is not None:
                    labels[i] = axis_labels[idx, :]
                value_idx = idx

            values = values[(slice(None),) * i + (value_idx,)]

        return type(self)(tuple(labels), values)

    equals = binary.equals
    pivot = pivot
    unpivot = unpivot

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

    # Matrix multiplication
    __matmul__ = binary.matmul

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


SomeEagerFrameLabeledArray = TypeVar(
    "SomeEagerFrameLabeledArray", bound=EagerFrameLabeledArray[AnyDataFrame]
)

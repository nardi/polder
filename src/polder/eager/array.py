from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import Self, TypeVar, cast, overload

from array_api_compat import array_namespace
from narwhals import Expr

import polder.eager.binary as binary
import polder.eager.unary as unary
from polder.eager.labels import Labels
from polder.eager.pivot import pivot, unpivot
from polder.eager.value_array import (
    AnyValueArray,
    SomeValueArray,
    ValueArrayNamespace,
    array_equal,
)
from polder.protocols.array import (
    AnyDataFrame,
    ArrayAxisIndices,
    AxisIndices,
    FrameLabeledArray,
    LabelFrameType,
)


def swap_args(f):
    return lambda a, b: f(b, a)


@dataclass(frozen=True, eq=False, slots=True)
class EagerFrameLabeledArray(FrameLabeledArray[LabelFrameType, SomeValueArray]):
    _labels: Labels[LabelFrameType]
    _values: SomeValueArray

    @classmethod
    def create(
        cls, labels: Sequence[LabelFrameType | None], values: SomeValueArray
    ) -> Self:
        return cls(Labels(labels), values)

    def __post_init__(self):
        # Check that we have the correct label container.
        assert isinstance(self._labels, Labels)
        # Check that the value array's shape is determined.
        assert all(isinstance(n, int) for n in self._values.shape)
        # Check that the labels and values have the same shape.
        labels_shape = tuple(
            len(labels) if labels is not None else 1 for labels in self._labels
        )
        assert labels_shape == self._values.shape

    @cached_property
    def array_namespace(self) -> ValueArrayNamespace[SomeValueArray]:
        return array_namespace(self._values)

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

    def values(self, *indices: ArrayAxisIndices) -> SomeValueArray:
        return self._values[indices or ...]

    def shape(self) -> tuple[int, ...]:
        # The validity of the shape is checked during __post_init__.
        return cast(tuple[int, ...], self._values.shape)

    def __getitem__(self, indices: AxisIndices | tuple[AxisIndices, ...]) -> Self:
        xp = self.array_namespace

        if not isinstance(indices, tuple):
            indices = (indices,)

        labels = list(self._labels)
        values = self._values

        # Index each axis one-by-one.
        n_removed_axes = 0
        for i, idx in enumerate(indices):
            # If we have removed some axes, later ones will shift to the left,
            # so we have to compensate for this in the indexing.
            j = i - n_removed_axes

            # If the index is None, we want to add an unlabeled axis.
            if idx is None:
                labels.insert(j, None)
                value_idx = None

            # If we are indexing with a mapping, filter the axis labels and
            # remove the filtered columns. Then use the result to index the
            # values.
            elif isinstance(idx, Mapping):
                axis_labels = labels[j]
                if axis_labels is None:
                    raise Exception("Cannot index unlabeled index with a mapping")
                filtered_axis_labels = axis_labels.with_row_index(
                    "__value_index"
                ).filter(**idx)
                unfiltered_columns = [
                    col for col in axis_labels.columns if col not in idx
                ]
                if unfiltered_columns:
                    labels[j] = filtered_axis_labels.select(unfiltered_columns)
                    value_idx = xp.asarray(
                        filtered_axis_labels["__value_index"].to_numpy()
                    )
                else:
                    # If there are no unfiltered columns left, remove the axis
                    # entirely.
                    labels.pop(j)
                    n_removed_axes += 1
                    value_idx = filtered_axis_labels["__value_index"].item()

            # If we are indexing with an Expr, filter the axis labels and then
            # use the result to index the values.
            elif isinstance(idx, Expr):
                axis_labels = labels[j]
                if axis_labels is None:
                    raise Exception("Cannot index unlabeled index with an expression")
                filtered_axis_labels = axis_labels.with_row_index(
                    "__value_index"
                ).filter(idx)
                labels[j] = filtered_axis_labels.select(axis_labels.columns)
                value_idx = xp.asarray(filtered_axis_labels["__value_index"].to_numpy())

            # If we have a single-valued numerical indexer, indexing the values
            # will lower the array by 1 dimension, so remove the axis in the
            # labels as well.
            elif isinstance(idx, int):
                labels.pop(j)
                n_removed_axes += 1
                value_idx = idx

            # If we have a numerical indexer (int array or slice) we index
            # both labels and values with the indexer.
            else:
                axis_labels = labels[j]
                if axis_labels is None:
                    # For unlabeled axes, we only allow indexing with `:`.
                    idx_is_full_slice = isinstance(idx, slice) and idx == slice(None)
                    if not idx_is_full_slice:
                        raise Exception(
                            "Cannot index unlabeled index with multi-valued indexer"
                        )
                else:
                    labels[j] = axis_labels[idx, :]
                value_idx = idx

            values = values[(slice(None),) * j + (value_idx,)]

        return self.create(labels, values)

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


# If JAX is installed, register as pytree.
try:
    from jax.tree_util import register_dataclass

    register_dataclass(
        EagerFrameLabeledArray, data_fields=["_values"], meta_fields=["_labels"]
    )
except ImportError:
    pass


SomeEagerFrameLabeledArray = TypeVar(
    "SomeEagerFrameLabeledArray",
    bound=EagerFrameLabeledArray[AnyDataFrame, AnyValueArray],
)

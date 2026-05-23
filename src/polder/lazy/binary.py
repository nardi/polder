from __future__ import annotations

import operator
from collections.abc import Callable
from typing import TYPE_CHECKING, TypeAlias

import narwhals as nw
import numpy as np

from polder.lazy.align import align

if TYPE_CHECKING:
    from polder.lazy.array import SomeLazyFrameLabeledArray

Scalar: TypeAlias = int | float | complex | bool


def _generate_binop(op: Callable[[nw.Expr, nw.Expr], nw.Expr]):
    def _perform_binop(
        left: SomeLazyFrameLabeledArray | Scalar,
        right: SomeLazyFrameLabeledArray | Scalar,
        /,
    ) -> SomeLazyFrameLabeledArray:
        from polder.lazy.array import LazyFrameLabeledArray

        def upcast_scalar(
            ref_array: SomeLazyFrameLabeledArray, scalar: Scalar
        ) -> SomeLazyFrameLabeledArray:
            frame_ns = ref_array._frame_ns
            n_dims = ref_array._n_dims
            indexed_labels = tuple(None for _ in ref_array._indexed_labels)
            values = ref_array.maybe_lazy(
                nw.DataFrame.from_dict(
                    {**{f"__index{i}": 0 for i in range(n_dims)}, "value": scalar},
                    backend=frame_ns,
                )
            )
            shape = ref_array.maybe_lazy(
                nw.DataFrame.from_dict(
                    {"axis": np.arange(n_dims), "size": 1},
                    backend=frame_ns,
                )
            )
            return type(ref_array)(indexed_labels, values, shape, n_dims, frame_ns)

        if isinstance(left, LazyFrameLabeledArray) and isinstance(
            right, LazyFrameLabeledArray
        ):
            left_array = left
            right_array = right
        elif isinstance(left, LazyFrameLabeledArray) and isinstance(right, Scalar):
            left_array = left
            right_array = upcast_scalar(left, right)
        elif isinstance(right, LazyFrameLabeledArray) and isinstance(left, Scalar):
            left_array = upcast_scalar(right, left)
            right_array = right
        else:
            raise NotImplementedError(
                "Cannot perform binary array operation without any arrays."
            )

        left_array, right_array = align(left_array, right_array)

        frame_ns = left_array._frame_ns
        n_dims = left_array._n_dims

        indexed_labels = tuple(
            l1 if l1 is not None else l2
            for l1, l2 in zip(
                left_array._indexed_labels, right_array._indexed_labels, strict=True
            )
        )

        shape = left_array._shape.join(
            right_array._shape,  # type: ignore
            on="axis",
            how="inner",
        ).select("axis", nw.max_horizontal("size", "size_right").alias("size"))

        # Keep track of which axes are unlabeled and will be broadcasted, and
        # exclude those from the join columns.
        unlabeled_axes = [
            (l1 is None, l2 is None)
            for l1, l2 in zip(
                left_array._indexed_labels, right_array._indexed_labels, strict=True
            )
        ]
        broadcasted_axes = [
            i
            for i, (left_unlabeled, right_unlabeled) in enumerate(unlabeled_axes)
            if left_unlabeled or right_unlabeled
        ]
        aligned_axis_columns = [
            f"__index{i}" for i in range(n_dims) if i not in broadcasted_axes
        ]

        # Combine the values in both arrays.
        if aligned_axis_columns:
            values = left_array._values.join(
                right_array._values,  # type: ignore
                on=aligned_axis_columns,
                how="inner",
            )
        else:
            values = left_array._values.join(
                right_array._values,  # type: ignore
                how="cross",
            )

        # Pick a single set of indices for the values.
        values = values.select(
            *[
                # For each axis, take the indices of the left array, unless that
                # is an unlabeled axis. In that case it is either broadcasted
                # over the right axis, or both are unlabeled. In both cases it
                # is then fine to pick the right indices.
                f"__index{i}"
                if not left_unlabeled
                else nw.col(f"__index{i}_right").alias(f"__index{i}")
                for i, (left_unlabeled, _) in enumerate(unlabeled_axes)
            ],
            # Apply the operator to the two value columns.
            op(nw.col("value"), nw.col("value_right")).alias("value"),
        )

        return type(left_array)(indexed_labels, values, shape, n_dims, frame_ns)  # type: ignore

    return _perform_binop


# Arithmetic operators
add = _generate_binop(operator.add)
sub = _generate_binop(operator.sub)
mul = _generate_binop(operator.mul)
truediv = _generate_binop(operator.truediv)
floordiv = _generate_binop(operator.floordiv)
mod = _generate_binop(operator.mod)
pow = _generate_binop(operator.pow)

# Bitwise operators
and_ = _generate_binop(operator.and_)
or_ = _generate_binop(operator.or_)
# Narwhals doesn't have xor, lshift and rshift, so we'll have to write them out.
xor = _generate_binop(lambda a, b: (a | b) & ~(a & b))
lshift = _generate_binop(lambda a, b: a * 2**b)
rshift = _generate_binop(lambda a, b: a // 2**b)

# Comparison operators
lt = _generate_binop(operator.lt)
le = _generate_binop(operator.le)
gt = _generate_binop(operator.gt)
ge = _generate_binop(operator.ge)
eq = _generate_binop(operator.eq)
ne = _generate_binop(operator.ne)

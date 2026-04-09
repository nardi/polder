from __future__ import annotations

import operator
from collections.abc import Callable
from typing import TYPE_CHECKING, TypeAlias

import narwhals as nw
import narwhals.typing as nwt
import numpy as np

from polder.eager.align import align

if TYPE_CHECKING:
    from polder.eager.array import SomeEagerFrameLabeledArray

__all__ = [
    "equals",
    "add",
    "sub",
    "mul",
    "truediv",
    "floordiv",
    "mod",
    "pow",
    "and_",
    "or_",
    "xor",
    "lshift",
    "rshift",
    "lt",
    "le",
    "gt",
    "ge",
    "eq",
    "ne",
]


def _narwhals_df_equals(l1: nwt.DataFrameT, l2: nwt.DataFrameT) -> bool:
    """Determines equality of two DataFrames. Considers them equal if they have the same type,
    columns and rows, with ordering for both being the same as well."""
    # Two DataFrames are not equal if they have different types, different columns, or a different
    # number of rows.
    if type(l1) is not type(l2) or l1.columns != l2.columns or len(l1) != len(l2):
        return False

    # Otherwise, they are equal iff an outer join on all columns including row index creates no
    # extra rows.
    assert "__index" not in l1.columns
    return len(l1) == (
        l1.with_row_index("__index")
        .lazy()
        .join(
            l2.with_row_index("__index").lazy(), on=[*l1.columns, "__index"], how="full"
        )
        .select(nw.col("__index").fill_null(-1).count())
        .collect()
        .item()
    )


def equals(a: SomeEagerFrameLabeledArray, b: SomeEagerFrameLabeledArray) -> bool:
    if type(a) is not type(b):
        return False

    return np.array_equal(a._values, b._values, equal_nan=True) and all(
        _narwhals_df_equals(l1, l2) if l1 is not None and l2 is not None else l1 is l2
        for l1, l2 in zip(a._labels, b._labels, strict=True)
    )


Scalar: TypeAlias = int | float | complex | np.generic


def _generate_binop(op: Callable):
    def _perform_binop(
        left: SomeEagerFrameLabeledArray | Scalar,
        right: SomeEagerFrameLabeledArray | Scalar,
    ) -> SomeEagerFrameLabeledArray:
        from polder.eager.array import EagerFrameLabeledArray

        def upcast_scalar(ref_array: SomeEagerFrameLabeledArray, scalar: Scalar):
            labels = tuple(None for _ in ref_array._labels)
            values = np.full((1,) * len(labels), scalar)
            return type(ref_array)(labels, values)

        if isinstance(left, EagerFrameLabeledArray) and isinstance(
            right, EagerFrameLabeledArray
        ):
            left_array = left
            right_array = right
        elif isinstance(left, EagerFrameLabeledArray) and isinstance(right, Scalar):
            left_array = left
            right_array = upcast_scalar(left, right)
        elif isinstance(right, EagerFrameLabeledArray) and isinstance(left, Scalar):
            left_array = upcast_scalar(right, left)
            right_array = right
        else:
            raise NotImplementedError(
                "Cannot perform binary array operation without any arrays."
            )

        left_array, right_array = align(left_array, right_array)  # type: ignore[assignment]
        labels = tuple(
            l1 if l1 is not None else l2
            for l1, l2 in zip(left_array._labels, right_array._labels, strict=True)
        )
        values = op(left_array._values, right_array._values)
        return type(left_array)(labels, values)

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
xor = _generate_binop(operator.xor)
lshift = _generate_binop(operator.lshift)
rshift = _generate_binop(operator.rshift)

# Comparison operators
lt = _generate_binop(operator.lt)
le = _generate_binop(operator.le)
gt = _generate_binop(operator.gt)
ge = _generate_binop(operator.ge)
eq = _generate_binop(operator.eq)
ne = _generate_binop(operator.ne)

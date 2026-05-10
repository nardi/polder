from __future__ import annotations

import operator
from collections.abc import Callable
from typing import TYPE_CHECKING, TypeAlias

from polder.config import auto_align
from polder.eager._narwhals_df_equals import narwhals_df_equals
from polder.eager.align import align
from polder.eager.value_array import array_equal

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


def equals(a: SomeEagerFrameLabeledArray, b: SomeEagerFrameLabeledArray) -> bool:
    if type(a) is not type(b):
        return False

    return array_equal(a._values, b._values, equal_nan=True) and all(
        narwhals_df_equals(l1, l2) if l1 is not None and l2 is not None else l1 is l2
        for l1, l2 in zip(a._labels, b._labels, strict=True)
    )


Scalar: TypeAlias = int | float | complex | bool


def _generate_binop(op: Callable):
    def _perform_binop(
        left: SomeEagerFrameLabeledArray | Scalar,
        right: SomeEagerFrameLabeledArray | Scalar,
        /,
    ) -> SomeEagerFrameLabeledArray:
        from polder.eager.array import EagerFrameLabeledArray

        def upcast_scalar(ref_array: SomeEagerFrameLabeledArray, scalar: Scalar):
            xp = ref_array.array_namespace
            labels = tuple(None for _ in ref_array._labels)
            values = xp.full((1,) * len(labels), scalar)
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

        # Use the auto_align setting to determine whether to perform alignment or just check it
        should_align = auto_align()
        left_array, right_array = align(
            left_array, right_array, check_only=not should_align
        )  # type: ignore[assignment]
        labels = tuple(
            l1 if l1 is not None else l2
            for l1, l2 in zip(left_array._labels, right_array._labels, strict=True)
        )
        values = op(left_array._values, right_array._values)
        return type(left_array)(labels, values)

    return _perform_binop


def matmul(
    left: SomeEagerFrameLabeledArray,
    right: SomeEagerFrameLabeledArray,
) -> SomeEagerFrameLabeledArray:
    """Matrix multiplication of two labeled arrays.

    Performs matrix multiplication on the values and handles labels appropriately.
    For 2D arrays: (m, n) @ (n, p) -> (m, p)
    The result has the labels from left's first axis and right's second axis.
    """
    # Align the arrays.
    left, right = align(left, right, axes=((-1, 0),))

    # Perform matrix multiplication on values.
    result_values = left._values @ right._values

    # Construct result labels: all but last from left, all but first from right.
    result_labels = tuple(list(left._labels[:-1]) + list(right._labels[1:]))

    return type(left)(result_labels, result_values)


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

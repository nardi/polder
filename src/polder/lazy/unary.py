from __future__ import annotations

import operator
from collections.abc import Callable
from typing import TYPE_CHECKING

import narwhals as nw
from narwhals import Expr

if TYPE_CHECKING:
    from polder.lazy.array import SomeLazyFrameLabeledArray

__all__ = [
    "pos",
    "neg",
    "abs_",
    "invert",
    "bitwise_invert",
    "ceil",
    "cos",
    "exp",
    "expm1",
    "floor",
    "isfinite",
    "isnan",
    "log",
    "log2",
    "log10",
    "logical_not",
    "reciprocal",
    "round_",
    "signbit",
    "sin",
    "sqrt",
    "square",
]


def _generate_unop(
    op: Callable[[nw.Expr], nw.Expr],
):
    """Generates a unary operation by applying a Narwhals expression
    transformation to the `value` column of the lazy array's `values` frame."""

    def _perform_unop(arr: SomeLazyFrameLabeledArray, /) -> SomeLazyFrameLabeledArray:
        values = arr._values.with_columns(op(nw.col("value")).alias("value"))
        return type(arr)(
            arr._indexed_labels, values, arr._shape, arr._n_dims, arr._frame_ns
        )

    return _perform_unop


# Operators
# Narwhals Expr does not support unary + or -, so we use arithmetic for neg and
# the identity for pos.
pos = _generate_unop(lambda e: e)
neg = _generate_unop(lambda e: nw.lit(0) - e)
abs_ = _generate_unop(Expr.abs)
invert = _generate_unop(operator.invert)

# Trigonometric functions (only cos and sin are available in Narwhals Expr)
cos = _generate_unop(Expr.cos)
sin = _generate_unop(Expr.sin)

# Exponential and logarithmic functions
exp = _generate_unop(Expr.exp)
expm1 = _generate_unop(lambda e: e.exp() - 1)
log = _generate_unop(Expr.log)
log2 = _generate_unop(lambda e: e.log(2))
log10 = _generate_unop(lambda e: e.log(10))

# Rounding functions
ceil = _generate_unop(Expr.ceil)
floor = _generate_unop(Expr.floor)
round_ = _generate_unop(Expr.round)

# Reciprocal and power functions
reciprocal = _generate_unop(lambda e: nw.lit(1) / e)
sqrt = _generate_unop(Expr.sqrt)
square = _generate_unop(lambda e: e**2)

# Bitwise functions
bitwise_invert = _generate_unop(operator.invert)

# Logical and classification functions
isfinite = _generate_unop(Expr.is_finite)
isnan = _generate_unop(Expr.is_nan)
logical_not = _generate_unop(operator.invert)
signbit = _generate_unop(lambda e: e < 0)

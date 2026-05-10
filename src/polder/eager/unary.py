from __future__ import annotations

import operator
from collections.abc import Callable
from typing import TYPE_CHECKING

from polder.eager.value_array import SomeValueArray, ValueArrayNamespace
from polder.protocols.array import LabelFrameType

if TYPE_CHECKING:
    from polder.eager.array import EagerFrameLabeledArray

__all__ = [
    "pos",
    "neg",
    "abs_",
    "invert",
    "acos",
    "acosh",
    "asin",
    "asinh",
    "atan",
    "atanh",
    "bitwise_invert",
    "ceil",
    "conj",
    "cos",
    "cosh",
    "exp",
    "expm1",
    "floor",
    "imag",
    "isfinite",
    "isinf",
    "isnan",
    "log",
    "log1p",
    "log2",
    "log10",
    "logical_not",
    "real",
    "reciprocal",
    "round_",
    "sign",
    "signbit",
    "sin",
    "sinh",
    "square",
    "sqrt",
    "tan",
    "tanh",
    "trunc",
]


def _generate_unop(
    op_generator: Callable[
        [ValueArrayNamespace[SomeValueArray]],
        Callable[[SomeValueArray], SomeValueArray],
    ],
):
    """Generates a unary operation by selecting the corresponding operation from
    the value array namespace, and wrapping the result in the frame labeled
    array type."""

    def _perform_unop(
        arr: EagerFrameLabeledArray[LabelFrameType, SomeValueArray], /
    ) -> EagerFrameLabeledArray[LabelFrameType, SomeValueArray]:
        xp = arr.array_namespace
        op = op_generator(xp)
        values = op(arr._values)
        return type(arr)(arr._labels, values)

    return _perform_unop


# Operators
pos = _generate_unop(lambda _: operator.pos)
neg = _generate_unop(lambda _: operator.neg)
abs_ = _generate_unop(lambda _: operator.abs)
invert = _generate_unop(lambda _: operator.invert)

# Trigonometric functions
acos = _generate_unop(lambda xp: xp.acos)
asin = _generate_unop(lambda xp: xp.asin)
atan = _generate_unop(lambda xp: xp.atan)
cos = _generate_unop(lambda xp: xp.cos)
sin = _generate_unop(lambda xp: xp.sin)
tan = _generate_unop(lambda xp: xp.tan)

# Inverse hyperbolic functions
acosh = _generate_unop(lambda xp: xp.acosh)
asinh = _generate_unop(lambda xp: xp.asinh)
atanh = _generate_unop(lambda xp: xp.atanh)

# Hyperbolic functions
cosh = _generate_unop(lambda xp: xp.cosh)
sinh = _generate_unop(lambda xp: xp.sinh)
tanh = _generate_unop(lambda xp: xp.tanh)

# Exponential and logarithmic functions
exp = _generate_unop(lambda xp: xp.exp)
expm1 = _generate_unop(lambda xp: xp.expm1)
log = _generate_unop(lambda xp: xp.log)
log1p = _generate_unop(lambda xp: xp.log1p)
log2 = _generate_unop(lambda xp: xp.log2)
log10 = _generate_unop(lambda xp: xp.log10)

# Rounding functions
ceil = _generate_unop(lambda xp: xp.ceil)
floor = _generate_unop(lambda xp: xp.floor)
round_ = _generate_unop(lambda xp: xp.round)
trunc = _generate_unop(lambda xp: xp.trunc)

# Sign and reciprocal functions
reciprocal = _generate_unop(lambda xp: xp.reciprocal)
sign = _generate_unop(lambda xp: xp.sign)
sqrt = _generate_unop(lambda xp: xp.sqrt)
square = _generate_unop(lambda xp: xp.square)

# Bitwise functions
bitwise_invert = _generate_unop(lambda _: operator.invert)

# Complex functions
conj = _generate_unop(lambda xp: xp.conj)
imag = _generate_unop(lambda xp: xp.imag)
real = _generate_unop(lambda xp: xp.real)

# Logical and classification functions
isfinite = _generate_unop(lambda xp: xp.isfinite)
isinf = _generate_unop(lambda xp: xp.isinf)
isnan = _generate_unop(lambda xp: xp.isnan)
logical_not = _generate_unop(lambda xp: xp.logical_not)
signbit = _generate_unop(lambda xp: xp.signbit)

from __future__ import annotations

import operator
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from polder.eager.array import SomeEagerFrameLabeledArray

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


def _generate_unop(op):
    def _perform_unop(arr: SomeEagerFrameLabeledArray, /) -> SomeEagerFrameLabeledArray:
        values = op(arr._values)
        return type(arr)(arr._labels, values)

    return _perform_unop


# Operators
pos = _generate_unop(operator.pos)
neg = _generate_unop(operator.neg)
abs_ = _generate_unop(operator.abs)
invert = _generate_unop(operator.invert)

# Trigonometric functions
acos = _generate_unop(np.arccos)
asin = _generate_unop(np.arcsin)
atan = _generate_unop(np.arctan)
cos = _generate_unop(np.cos)
sin = _generate_unop(np.sin)
tan = _generate_unop(np.tan)

# Inverse hyperbolic functions
acosh = _generate_unop(np.arccosh)
asinh = _generate_unop(np.arcsinh)
atanh = _generate_unop(np.arctanh)

# Hyperbolic functions
cosh = _generate_unop(np.cosh)
sinh = _generate_unop(np.sinh)
tanh = _generate_unop(np.tanh)

# Exponential and logarithmic functions
exp = _generate_unop(np.exp)
expm1 = _generate_unop(np.expm1)
log = _generate_unop(np.log)
log1p = _generate_unop(np.log1p)
log2 = _generate_unop(np.log2)
log10 = _generate_unop(np.log10)

# Rounding functions
ceil = _generate_unop(np.ceil)
floor = _generate_unop(np.floor)
round_ = _generate_unop(np.round)
trunc = _generate_unop(np.trunc)

# Sign and reciprocal functions
reciprocal = _generate_unop(np.reciprocal)
sign = _generate_unop(np.sign)
sqrt = _generate_unop(np.sqrt)
square = _generate_unop(np.square)

# Bitwise functions
bitwise_invert = _generate_unop(operator.invert)

# Complex functions
conj = _generate_unop(np.conj)
imag = _generate_unop(np.imag)
real = _generate_unop(np.real)

# Logical and classification functions
isfinite = _generate_unop(np.isfinite)
isinf = _generate_unop(np.isinf)
isnan = _generate_unop(np.isnan)
logical_not = _generate_unop(np.logical_not)
signbit = _generate_unop(np.signbit)

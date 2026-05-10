"""Dispatching layer for unary operations based on array type."""

import polder.eager.unary as eager
from polder.eager.array import EagerFrameLabeledArray
from polder.protocols.array import SomeFrameLabeledArray

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


def _dispatch_unop(eager_op):
    def _dispatcher(arr: SomeFrameLabeledArray, /) -> SomeFrameLabeledArray:
        if isinstance(arr, EagerFrameLabeledArray):
            return eager_op(arr)
        raise NotImplementedError(f"Unary operation not implemented for {type(arr)}")

    return _dispatcher


# Create dispatchers for all unary operations.
pos = _dispatch_unop(eager.pos)
neg = _dispatch_unop(eager.neg)
abs_ = _dispatch_unop(eager.abs_)
invert = _dispatch_unop(eager.invert)
acos = _dispatch_unop(eager.acos)
acosh = _dispatch_unop(eager.acosh)
asin = _dispatch_unop(eager.asin)
asinh = _dispatch_unop(eager.asinh)
atan = _dispatch_unop(eager.atan)
atanh = _dispatch_unop(eager.atanh)
bitwise_invert = _dispatch_unop(eager.bitwise_invert)
ceil = _dispatch_unop(eager.ceil)
conj = _dispatch_unop(eager.conj)
cos = _dispatch_unop(eager.cos)
cosh = _dispatch_unop(eager.cosh)
exp = _dispatch_unop(eager.exp)
expm1 = _dispatch_unop(eager.expm1)
floor = _dispatch_unop(eager.floor)
imag = _dispatch_unop(eager.imag)
isfinite = _dispatch_unop(eager.isfinite)
isinf = _dispatch_unop(eager.isinf)
isnan = _dispatch_unop(eager.isnan)
log = _dispatch_unop(eager.log)
log1p = _dispatch_unop(eager.log1p)
log2 = _dispatch_unop(eager.log2)
log10 = _dispatch_unop(eager.log10)
logical_not = _dispatch_unop(eager.logical_not)
real = _dispatch_unop(eager.real)
reciprocal = _dispatch_unop(eager.reciprocal)
round_ = _dispatch_unop(eager.round_)
sign = _dispatch_unop(eager.sign)
signbit = _dispatch_unop(eager.signbit)
sin = _dispatch_unop(eager.sin)
sinh = _dispatch_unop(eager.sinh)
square = _dispatch_unop(eager.square)
sqrt = _dispatch_unop(eager.sqrt)
tan = _dispatch_unop(eager.tan)
tanh = _dispatch_unop(eager.tanh)
trunc = _dispatch_unop(eager.trunc)

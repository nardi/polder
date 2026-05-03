from __future__ import annotations

import operator
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from polder.eager.array import SomeEagerFrameLabeledArray

__all__ = [
    "pos",
    "neg",
    "abs_",
    "invert",
]


def _generate_unop(op):
    def _perform_unop(arr: SomeEagerFrameLabeledArray, /) -> SomeEagerFrameLabeledArray:
        values = op(arr._values)
        return type(arr)(arr._labels, values)

    return _perform_unop


pos = _generate_unop(operator.pos)
neg = _generate_unop(operator.neg)
abs_ = _generate_unop(operator.abs)
invert = _generate_unop(operator.invert)

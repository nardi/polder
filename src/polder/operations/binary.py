import polder.eager.binary as eager
from polder.eager.array import EagerFrameLabeledArray
from polder.protocols.array import SomeFrameLabeledArray


def equals(a: SomeFrameLabeledArray, b: SomeFrameLabeledArray) -> bool:
    if type(a) is not type(b):
        return False

    if isinstance(a, EagerFrameLabeledArray):
        return eager.equals(a, b)

    raise NotImplementedError()

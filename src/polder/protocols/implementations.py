from enum import StrEnum, auto


class FrameLabeledArrayImplementation(StrEnum):
    EAGER = auto()
    LAZY = auto()


EAGER = FrameLabeledArrayImplementation.EAGER
LAZY = FrameLabeledArrayImplementation.LAZY

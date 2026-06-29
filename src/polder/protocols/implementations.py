from enum import StrEnum, auto


class FrameLabeledArrayImplementation(StrEnum):
    """Selects which implementation of the protocol to build or convert to."""

    EAGER = auto()
    """The array-backed implementation that resolves every operation immediately."""
    LAZY = auto()
    """The DataFrame-backed implementation that evaluates operations lazily."""


EAGER = FrameLabeledArrayImplementation.EAGER
"""Shorthand for `FrameLabeledArrayImplementation.EAGER`."""
LAZY = FrameLabeledArrayImplementation.LAZY
"""Shorthand for `FrameLabeledArrayImplementation.LAZY`."""

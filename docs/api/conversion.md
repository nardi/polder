# Conversion

Moving an array between the eager and lazy implementations. See
[Converting between implementations](../guide/conversion.md) for a guide.

::: polder.operations.conversion.convert

## Implementation selection

The target implementation is chosen with the `FrameLabeledArrayImplementation` enum, whose
members `EAGER` and `LAZY` are re-exported at the top level as `pld.EAGER` and `pld.LAZY`.

::: polder.protocols.implementations.FrameLabeledArrayImplementation

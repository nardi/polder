# The protocol

`FrameLabeledArray` is the protocol that all implementations satisfy, and the type your own
code should be written against. See [Writing generic code](../guide/generic-protocol.md) for
an introduction.

::: polder.protocols.array.FrameLabeledArray

## Supporting types

`ValuesIndexer` is the type of the `values` attribute. It supports both being called and
being subscripted, which is what lets `array.values()` and `array.values[...]` both work.

::: polder.protocols.array.ValuesIndexer

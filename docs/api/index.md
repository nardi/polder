# API reference

This reference is generated from the source code and its docstrings. For an
explanation-first introduction, start with the [user guide](../guide/eager-arrays.md).

The recommended import is `import polder as pld`. The most commonly used names are
re-exported at the top level of the package, so you can reach them as `pld.<name>`.

## Module map

- [Creation](creation.md): build arrays from values and labels, or from a single frame.
- [Conversion](conversion.md): move an array between implementations.
- [The protocol](protocol.md): the `FrameLabeledArray` protocol that your code targets.
- [Eager implementation](eager.md): the array-backed `EagerFrameLabeledArray`.
- [Lazy implementation](lazy.md): the DataFrame-backed `LazyFrameLabeledArray`.
- [Operations](operations.md): unary elementwise functions and alignment.
- [Configuration](config.md): global settings such as automatic alignment.

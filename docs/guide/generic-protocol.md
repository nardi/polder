# Writing generic code

At the center of the library is the `FrameLabeledArray` protocol, and this is what your
code should be written against. The idea is that you write your computation once, against
the protocol, and then run it against whichever implementation suits the situation. The
same function can operate on an [eager array](eager-arrays.md) backed by NumPy or JAX, and
on a [lazy array](lazy-arrays.md) backed by a DataFrame engine.

## The protocol, not a base class

`FrameLabeledArray` is a [`typing.Protocol`](https://docs.python.org/3/library/typing.html#typing.Protocol),
not a base class that implementations inherit from. The implementations deliberately do not
share a common ancestor. Instead, each one implements the protocol independently and is
free to vary its internals however it needs to. This keeps the implementations free to
specialize for efficiency, while still presenting one interface to your code.

A consequence of this design is that the decision of what belongs in the protocol is
significant. Anything in the protocol has to generalize to every implementation, and it is
also what users write against, so the protocol is kept stable and deliberately small. It
mostly follows the [array API](https://data-apis.org/array-api/latest/), since a
frame-labeled array is an array first. Additional functionality such as
[alignment](alignment.md) and [pivoting](pivoting.md) is included where it makes sense for
the format, and some array API functionality is extended, such as
[indexing with an expression](indexing.md).

## Targeting the protocol

The protocol is generic over two type variables: the label frame type and the value array
type. To write a function that accepts any frame-labeled array, annotate against
`FrameLabeledArray`.

```python
import numpy as np
import polars as pl

import polder as pld
from polder.protocols.array import FrameLabeledArray


def normalize(array: FrameLabeledArray) -> FrameLabeledArray:
    """Scale the values so that the maximum magnitude is one."""
    return array / pld.abs_(array).values().max()


array = pld.from_values_and_labels(
    np.array([1.0, -4.0, 2.0]),
    [pl.DataFrame({"i": [0, 1, 2]})],
)
normalized = normalize(array)
assert normalized.values().max() == 0.5
```

When you want to express that a function returns the *same* implementation it was given,
use the `SomeFrameLabeledArray` type variable. This preserves the concrete type through
the call, so a tool such as a type checker knows that passing an eager array yields an
eager array.

```{.python continuation}
from polder.protocols.array import SomeFrameLabeledArray


def twice(array: SomeFrameLabeledArray) -> SomeFrameLabeledArray:
    return array + array


doubled = twice(array)
assert (doubled.values() == array.values() * 2).all()
```

## What you may assume

Because the protocol has to make sense for every backend, it only includes functionality
that is meaningful in every paradigm. A few assumptions are worth keeping in mind when you
write generic code:

- **Arrays are immutable.** Every operation returns a new array. Write in a functional
  style and do not expect in-place mutation.
- **Laziness varies.** The degree to which a computation is resolved immediately depends on
  the implementation, and the protocol does not make this explicit. You should expect a
  computation to be resolved as soon as the array values are converted to an eager form,
  for example when you call `array.values()`.
- **Not every implementation supports everything yet.** The lazy implementation does not
  yet implement the whole protocol. If your generic code uses a feature it lacks, you will
  get a `NotImplementedError`. You can [convert](conversion.md) to the eager implementation
  to get the full feature set.

Because the library relies so heavily on structural typing, all of its own code is fully
typed, and writing your generic code with type annotations will let your tools check that
it stays within the protocol.

## The shape of the protocol

The protocol exposes the value array through `values`, the labels through `labels`, and the
shape through `shape`. On top of that it defines [indexing](indexing.md),
[equality](alignment.md), [pivoting and unpivoting](pivoting.md), and the full set of
[array API operators](math-operations.md). The exact signatures are listed in the
[API reference](../api/protocol.md).

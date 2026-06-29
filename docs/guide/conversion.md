# Converting between implementations

One advantage of having more than one implementation of the same protocol is that you can
move between them. A natural workflow is to do some of the work fully in the relational
context, so that as much of the computation as possible can be handled by a single query
engine, and then switch over to the array-backed model when interoperation with an array
library becomes important.

The `convert` function moves an array to a different implementation.

```python
import numpy as np
import polars as pl

import polder as pld

values = np.arange(6.0).reshape(2, 3)
labels = [pl.DataFrame({"i": [0, 1]}), pl.DataFrame({"j": [0, 1, 2]})]

eager = pld.from_values_and_labels(values, labels, implementation=pld.EAGER)

# Move to the lazy implementation.
lazy = pld.convert(eager, implementation=pld.LAZY)

# And back again.
eager_again = pld.convert(lazy, implementation=pld.EAGER)
```

The target implementation is given by the `implementation` keyword, using the `EAGER` and
`LAZY` constants. It defaults to `EAGER`.

## No-op conversions are free

If the array is already in the target implementation, `convert` returns it unchanged
without copying any data. Because arrays are immutable this is always safe, and it means
you can call `convert` defensively without worrying about the cost.

```{.python continuation}
same = pld.convert(eager, implementation=pld.EAGER)
assert same is eager
```

## When to convert

Use the lazy implementation while you are still shaping and combining data that lives in a
DataFrame, so the backend's query engine can optimize the whole pipeline. Convert to the
eager implementation when you reach the numerical core of your computation, where you want
the values in a real array, need a feature the lazy implementation does not yet support
(such as [indexing](indexing.md) or [matrix multiplication](math-operations.md)), or want
to hand the array to an array library.

Converting from lazy to eager resolves the pending lazy computation, since it has to
produce concrete values. Converting from eager to lazy moves the values into the long,
relational format described in [Lazy arrays](lazy-arrays.md).

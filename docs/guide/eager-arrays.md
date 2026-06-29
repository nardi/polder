# Eager arrays

The eager implementation, `EagerFrameLabeledArray`, is the default and most complete
implementation of the [`FrameLabeledArray`](generic-protocol.md) protocol. It is "eager"
in the sense that every operation is resolved immediately, just as it would be with a
plain NumPy array. The values are stored in an actual array (following the
[array API](https://data-apis.org/array-api/latest/)), and the labels are stored as
[Narwhals](https://narwhals-dev.github.io/narwhals/) DataFrames.

This is the implementation you want when you need to interoperate with array libraries,
when you care about the way the values are laid out in memory, or when you want to perform
real numerical work on accelerators. If instead your data already lives in a DataFrame and
you would like to keep it there, see [Lazy arrays](lazy-arrays.md).

## Creating an eager array

The usual way to create an array is `from_values_and_labels`. You pass a value array and
one label frame per axis. The eager implementation is selected by default, so you do not
need to specify it.

```python
import numpy as np
import polars as pl

import polder as pld

values = np.array([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0],
])
labels = [
    pl.DataFrame({"city": ["Amsterdam", "Rotterdam"]}),
    pl.DataFrame({"year": [2021, 2022, 2023]}),
]

array = pld.from_values_and_labels(values, labels)
```

There is one label frame per axis, so `len(labels)` must equal the number of dimensions of
`values`. The number of rows of each label frame must match the size of the corresponding
axis. The frame for an axis can have more than one column, which lets you attach several
labels to the same axis (see [Pivoting](pivoting.md) for where this becomes useful).

If an axis has size 1, it may be left unlabeled by passing `None` for that axis. Such axes
are treated as broadcastable, which is explained in [Alignment](alignment.md).

## Decomposing an array

An eager array decomposes into its two parts, the values and the labels.

### Values

The underlying value array is reached through `values`. It can be called like a method to
return the whole array, or indexed to return a part of it.

```{.python continuation}
array.values()        # the full NumPy array
array.values()[0, 0]  # a single element of that array
array.values[0]       # the first row, possibly without copying
```

The difference between `array.values()[...]` and `array.values[...]` is that the latter
informs the array you are only interested in a subset of the values. For lazy arrays this can save some computational work, but for eager arrays this is usually equivalent.

`values` returns whatever array type you put in. If you constructed the array from a NumPy
array you get a NumPy array back, and if you constructed it from a JAX array you get a JAX
array back.

### Labels

The labels are reached through the `labels` method. Without arguments it returns the full
sequence of label frames, one per axis. With an integer it returns the frame for a single
axis, which is often more convenient.

```{.python continuation}
array.labels()    # a sequence with one frame (or None) per axis
array.labels()[0] # the frame for the first axis
array.labels(-1)  # the frame for the last axis
```

The frames are Narwhals DataFrames. To get back to a native frame (for example a Polars
DataFrame) use Narwhals' `to_native`, and to operate on them use the Narwhals API.

## Immutability

Arrays are immutable. Every operation returns a new array rather than modifying the
existing one. This is a deliberate choice that keeps the protocol uniform across backends,
including those (such as JAX) for which in-place mutation is not natural. In practice this
means you write code in a functional style:

```{.python continuation}
shifted = array + 1.0   # array itself is unchanged
```

## Supported backends

The eager implementation works in principle with any array that follows the array API.
There is tested, fully typed support for [NumPy](https://numpy.org/) and
[JAX](https://jax.readthedocs.io/). JAX is an optional dependency, so you have to install
it yourself, but if you are passing in JAX arrays you will already have done so.

Because the values are a genuine array, the eager implementation cooperates with the tools
of its backend. For example, an `EagerFrameLabeledArray` is registered as a JAX pytree, so
you can pass one through `jax.jit` and differentiate through a labeled computation.

```{.python continuation}
import jax
import jax.numpy as jnp

array = pld.from_values_and_labels(jnp.arange(6.0).reshape(2, 3), labels)

@jax.jit
def f(a):
    return a * 2

result = f(array)   # tracing works through the labels
```

## What you can do next

Everything in the rest of the user guide applies to eager arrays:

- [Indexing](indexing.md) to select parts of the array.
- [Alignment](alignment.md) to combine arrays whose labels are in a different order.
- [Pivoting](pivoting.md) to reshape along label columns.
- [Mathematical operations](math-operations.md) for elementwise and matrix operations.

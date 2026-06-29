# polder

*Finding the middle ground between array and relational data models.*

`polder` is a Python package that provides a simple composite data container type:

1. a NumPy-style ([array API](https://data-apis.org/array-api/latest/)) array,
2. whose entries are labeled using Polars-compatible ([Narwhals](https://narwhals-dev.github.io/narwhals/)) DataFrames.

In other words, it lets you bundle your *data* and your *metadata* into a single
container, using an array-based library to handle the data, and a relational library to
handle the metadata.

## Why?

The common situation in which this is useful is when you have a NumPy(-like) array, but
find yourself having difficulties keeping track of what the values represent. You might
end up creating an auxiliary data structure for this, where you keep some list of labels
next to your array, so that you avoid (for example) combining two arrays with the same
values but in a different order. Then you might find yourself writing extra code to keep
track of those labels throughout your computation. `polder` aims to provide this data
structure in a more general way, that you can adapt to the needs of your application.

This is similar to the [pandas](https://pandas.pydata.org/) idea of a DataFrame, as a
NumPy array with labels. We are of the opinion that the relational model of the DataFrame
as promoted by Polars and similar libraries is the better fit, and so this library aims
to maintain a clear separation between the array part and the relational part. If you
have used [Xarray](https://docs.xarray.dev/en/stable/), the container will feel familiar,
but `polder` differs in scope. A few design goals capture the difference:

1. `polder` aims to be purely an orchestration layer over existing array and data
   processing libraries, so that all significant processing work is dispatched to those.
2. `polder` aims to provide a simple data model, so that you can convert array-based code
   without having to reconsider your data modelling approach.
3. `polder` aims to support a larger range of array and DataFrame backends, so it can be
   used in many contexts (for example, adding structure to a GPU-backed algorithm, or
   performing automatic differentiation through a labeled computation).
4. `polder` aims to avoid performance surprises compared to the backend libraries.

## Installation

```bash
pip install polder
```

`polder` depends only on [NumPy](https://numpy.org/) and
[Narwhals](https://narwhals-dev.github.io/narwhals/). To actually create labels you will
want a Narwhals-supported DataFrame library such as [Polars](https://pola.rs/), and to
use a backend other than NumPy (such as [JAX](https://jax.readthedocs.io/)) you have to
install it yourself. These backends are never required for end users, they are always
imported dynamically.

## A first example

The recommended way to import the library is `import polder as pld`, similar to other
array libraries.

```python
import numpy as np
import polars as pl

import polder as pld

# Some values, and a label frame for each axis.
values = np.array([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0],
])
labels = [
    pl.DataFrame({"city": ["Amsterdam", "Rotterdam"]}),
    pl.DataFrame({"year": [2021, 2022, 2023]}),
]

array = pld.from_values_and_labels(values, labels)

# The array behaves like a NumPy array.
doubled = array * 2

# But it carries its labels along.
assert doubled.shape() == (2, 3)
assert doubled.labels(0).columns == ["city"]   # the "city" frame
np.testing.assert_array_equal(doubled.values(), values * 2)
```

The value array is reachable through [`values`](guide/eager-arrays.md), the labels
through [`labels`](guide/eager-arrays.md), and most array operations are supported
directly on the container.

## Where to next

If you are new to the library, the user guide is meant to be read roughly in order:

- [Eager arrays](guide/eager-arrays.md) introduces the default implementation.
- [Lazy arrays](guide/lazy-arrays.md) covers the DataFrame-backed implementation.
- [Converting between implementations](guide/conversion.md) shows how to move between them.
- [Writing generic code](guide/generic-protocol.md) explains the protocol your code
  should target.
- [Indexing](guide/indexing.md), [Alignment](guide/alignment.md),
  [Pivoting](guide/pivoting.md), and [Mathematical operations](guide/math-operations.md)
  each cover one area of functionality in depth.

For exact signatures and the full list of operations, see the
[API reference](api/index.md).

## Status

`polder` is in early development. At the center of the library is the
[`FrameLabeledArray`](api/protocol.md) protocol, which is what user code should be written
against. There are currently two implementations of this protocol, an eager one and a
lazy one. The eager implementation supports the full protocol, while the lazy
implementation is still being filled in. Each page in the user guide notes where
functionality is not yet available.

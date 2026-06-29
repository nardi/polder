# Lazy arrays

The lazy implementation, `LazyFrameLabeledArray`, is backed entirely by
[Narwhals](https://narwhals-dev.github.io/narwhals/) LazyFrames and evaluates all
operations lazily. Where the [eager implementation](eager-arrays.md) stores its values in
a real array, the lazy implementation stores everything (both values and labels) in a
relational, long format, where there is one row per array element.

This is useful when you want to express your computation in an array style but do not need
any special interoperation with array libraries, and either do not care how the values are
stored or already have your data in a DataFrame format supported by Narwhals. By staying in
that format, you keep your data in the same backend it already lives in, and you let that
backend's query engine optimize large computations.

!!! note "Early development"

    The lazy implementation is still being built out and does not yet support the whole
    protocol. The places where a method is not yet available are called out below. For the
    full feature set today, use the eager implementation.

## Creating a lazy array

There are two ways to create a lazy array.

### From values and labels

You can use the same `from_values_and_labels` function as for eager arrays, but pass
`implementation=pld.LAZY`.

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

array = pld.from_values_and_labels(values, labels, implementation=pld.LAZY)
```

### From a single frame

If your data already lives in a single DataFrame in long format, `from_frame` turns it
into a one-dimensional lazy array directly. One column holds the values and the remaining
columns become the labels. The value column is called `"value"` by default, which you can
override with `value_column`.

```{.python continuation}
import narwhals as nw

frame = pl.DataFrame({
    "city": ["Amsterdam", "Rotterdam", "Utrecht"],
    "value": [1.0, 2.0, 3.0],
})

array = pld.from_frame(nw.from_native(frame), value_column="value")
```

`from_frame` produces a lazy array by default, since keeping the data in its DataFrame
backend is the whole point. An array created this way is always one-dimensional. To get a
higher-dimensional array, use `from_values_and_labels`, or reshape with
[Pivoting](pivoting.md).

## Resolving lazy computations

Because the implementation is lazy, an operation does not necessarily do any work when you
call it. The work happens when you extract concrete results. As a rule, you should expect a
computation to be resolved as soon as the array values are converted to an eager form, for
example when you call `array.values()`.

```{.python continuation}
result = array * 2               # no work done yet
numpy_values = result.values()   # the computation runs here
```

`values()` returns a NumPy array regardless of the underlying DataFrame backend, by
collecting the lazy frames and reshaping the result.

If you only require a subset of the values, you can also using indexing syntax on `values` directly, which may allow for skipping some calculations:

```{.python continuation}
last_value = result.values[-1]   # only needs the final value, may skip calculating the others
```

You can also force the underlying frames to be collected without leaving the lazy
implementation, using `collect`. This resolves the pending computations and stores the
results in a new lazy array, which is useful when you want to materialize an intermediate
result.

```{.python continuation}
materialized = result.collect()
```

## Eager evaluation for debugging

Lazy evaluation can make errors surface far from where they originate. For debugging it can
help to evaluate eagerly, using DataFrames instead of LazyFrames internally. The
[`config.use_eager_evaluation_for_lazy_arrays`](../api/config.md) setting controls this.

```{.python continuation}
with pld.config.use_eager_evaluation_for_lazy_arrays(True):
    array = pld.from_values_and_labels(values, labels, implementation=pld.LAZY)
    # Errors now surface closer to the operation that caused them.
```

Note that some errors will still only surface lazily. For example, an invalid-shape error
may only arise when the shape is extracted, not on the operation that produced it.

## What is supported today

The lazy implementation supports decomposition (`values`, `labels`, `shape`), the unary and
binary [mathematical operations](math-operations.md), and [pivoting](pivoting.md).

The following protocol members currently raise `NotImplementedError` on lazy arrays:

- [Indexing](indexing.md) with `__getitem__`.
- Equality checking with `equals`.
- Matrix multiplication with `@`.

If you need any of these, [convert](conversion.md) to the eager implementation first.

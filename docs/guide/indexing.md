# Indexing

Indexing a frame-labeled array follows NumPy-style indexing, and extends it so that you can
also select along an axis by filtering its labels. The two styles can be mixed in a single
indexing expression, one entry per axis, just as with a NumPy array.

!!! note "Eager only for now"

    Indexing is currently supported on [eager arrays](eager-arrays.md). To index a
    lazy array, [convert](conversion.md) it to the eager implementation first.

Given an N-dimensional array, there are five ways to index it.

## 1. Single-valued numerical indexing

When an axis is indexed with a single integer, the N-1 dimensional slice at that location
is returned, and that axis disappears from the result. This matches NumPy exactly.

```python
import numpy as np
import polars as pl

import polder as pld

values = np.array([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0],
    [7.0, 8.0, 9.0],
    [10.0, 11.0, 12.0],
    [13.0, 14.0, 15.0],
])
labels = [pl.DataFrame({"x": [0, 1, 2, 3, 4]}), pl.DataFrame({"y": [10, 20, 30]})]
array = pld.from_values_and_labels(values, labels)

row = array[2]  # shape (3,), the "x" axis is gone
column = array[:, 1]  # shape (5,), the "y" axis is gone

assert row.shape() == (3,)
assert column.shape() == (5,)
```

## 2. Multi-valued numerical indexing

When an axis is indexed with a slice or a numerical array (of the same type as the values),
the array is sliced along that axis and the result keeps its dimensionality. Indexing
several axes at once works as you would expect.

```{.python continuation}
sliced = array[1:3]              # shape (2, 3)
picked = array[[1, 3], [0, 2]]   # shape (2, 2)

assert sliced.shape() == (2, 3)
assert picked.shape() == (2, 2)
```

As with NumPy, a slice such as `array[1:3]` can return a view of the values, while indexing
with an array such as `array[[1, 3]]` returns a copy. The labels are sliced to match.

## 3. Simple label filtering

When an axis is indexed with a mapping (such as a `dict`), it is used to filter the labels
of that axis. The keys refer to columns in the label frame, and the values to the single
value that is kept for that column. After filtering, that column is removed from the label
frame, analogous to single-valued numerical indexing. If it was the only column, the whole
axis is removed and the result is N-1 dimensional.

```{.python continuation}
values = np.array([
    [1.0, 2.0, 3.0, 4.0],
    [5.0, 6.0, 7.0, 8.0],
    [9.0, 10.0, 11.0, 12.0],
])
labels = [
    pl.DataFrame({"x": ["a", "b", "c"]}),
    pl.DataFrame({"y": [10, 20, 30, 40]}),
]
array = pld.from_values_and_labels(values, labels)

row = array[{"x": "b"}, :]   # shape (4,), keeps the second row
col = array[:, {"y": 20}]    # shape (3,), keeps the second column

assert row.shape() == (4,)
assert col.shape() == (3,)
```

If the axis has several label columns, filtering on one of them keeps the axis but narrows
it. The filtered column is removed, and the others remain.

```{.python continuation}
labels = [
    pl.DataFrame({"letter": ["a", "b", "c"], "number": [1, 2, 3]}),
    pl.DataFrame({"y": [10, 20, 30]}),
]
array = pld.from_values_and_labels(values[:, :3], labels)

result = array[{"number": 2}, :]  # shape (1, 3), "letter" column kept
assert result.shape() == (1, 3)
```

## 4. Complex label filtering

When an axis is indexed with a [Narwhals](https://narwhals-dev.github.io/narwhals/)
expression, that expression is used to `filter` the label frame for that axis. Neither the
width of the label frame nor the dimensionality of the array changes, the axis is just made
shorter to keep the rows that match.

```{.python continuation}
import narwhals as nw

prices = pld.from_values_and_labels(
    np.arange(15.0).reshape(5, 3),
    [pl.DataFrame({"x": [0, 1, 2, 3, 4]}), pl.DataFrame({"y": [10, 20, 30]})],
)

# Keep only the rows of the first axis where x is greater than 2.
result = prices[nw.col("x") > 2]
assert result.shape() == (2, 3)
```

This is the most flexible form, since the full Narwhals expression API is available to
describe which rows to keep.

## 5. Axis creation indexing

When `None` is used to index the array at position `i`, a new size-1 unlabeled axis is
created between the current axes `i-1` and `i`. This is useful for broadcasting, exactly as
`None` (or `np.newaxis`) is in NumPy.

```{.python continuation}
a = pld.from_values_and_labels(np.arange(4.0), [pl.DataFrame({"i": np.arange(4)})])
b = pld.from_values_and_labels(np.arange(3.0), [pl.DataFrame({"j": np.arange(3)})])

# Outer product through broadcasting.
outer = a[:, None] * b[None, :]
assert outer.shape() == (4, 3)
```

The new axis is unlabeled (its label frame is `None`), which marks it as broadcastable. See
[Alignment](alignment.md) for how unlabeled axes interact with binary operations.

## Indexing the values directly

The five forms above index the whole container, labels included. If you only want to index
the underlying value array, and possibly avoid some work, use `values` with a subscript, as
in `array.values[...]`.

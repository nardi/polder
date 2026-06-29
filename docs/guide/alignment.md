# Alignment

The whole point of labeling an array is to keep track of what its values represent. So when
you combine two arrays, it is not enough for their shapes to match, their labels have to
match too. Otherwise the result is not meaningful. *Alignment* is the operation that
reorders arrays so that their labels line up.

## Automatic alignment

By default, binary operations align their operands for you. If you add two arrays whose
labels are in a different order, `polder` first reorders them so that the labels match, and
then performs the addition. You do not have to do anything.

```python
import numpy as np
import polars as pl

import polder as pld

values = np.array([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0],
    [7.0, 8.0, 9.0],
])
labels = [pl.DataFrame({"x": [0, 1, 2]}), pl.DataFrame({"y": [10, 20, 30]})]
array = pld.from_values_and_labels(values, labels)

# The same data, but with the first axis in the opposite order. Both the values
# and the "x" labels are reversed, so this is the same logical array.
shuffled = pld.from_values_and_labels(
    values[::-1],
    [pl.DataFrame({"x": [2, 1, 0]}), pl.DataFrame({"y": [10, 20, 30]})],
)

# Aligned automatically, then added.
result = array + shuffled
assert result.shape() == (3, 3)
np.testing.assert_array_equal(result.values(), array.values() * 2)
```

This is the one place where the library departs from the rule that operations have no
performance surprises compared to the backend, because alignment may have to reorder data.
It is on by default because an unaligned binary operation is almost always a mistake.

## Aligning explicitly

You can also align arrays yourself with the `align` function, which returns the arrays
reordered so that their labels are identical along every axis. This is useful when you want
to align once and then perform several operations, or when you want to inspect the aligned
arrays.

```{.python continuation}
from polder.operations.align import align

aligned1, aligned2 = align(array, shuffled)
assert aligned1.equals(aligned2)   # same data, now in the same order
```

`align` accepts any number of arrays. With the `axes` keyword you can restrict alignment to
particular axes, and with `check_only=True` you can verify that the arrays are already
aligned without reordering anything.

## The alignment rules

Alignment follows a fixed set of rules:

1. All arrays must have the same number of axes.
2. Unlabeled (size-1) axes are already aligned, since they will be broadcast.
3. Scalars are already aligned, since they will be broadcast.
4. All label frames for a single axis are aligned by reordering, so they must have the same
   length.
5. All label frames for a single axis must have the same columns.
6. After alignment, all label frames for each axis are identical, unless they are `None`.

In other words, two axes can be aligned when they carry the same set of labels, possibly in
a different order. If the columns differ, or the lengths differ, the arrays cannot be
aligned and an error is raised.

## Broadcasting with unlabeled axes

A size-1 axis may be left unlabeled, with `None` for its label frame. Such an axis is
treated as broadcastable, and it aligns against any labeled axis without reordering. This is
how you combine arrays of different shapes, by introducing unlabeled axes with
[axis creation indexing](indexing.md).

```{.python continuation}
a = pld.from_values_and_labels(np.arange(4.0), [pl.DataFrame({"i": np.arange(4)})])
b = pld.from_values_and_labels(np.arange(3.0), [pl.DataFrame({"j": np.arange(3)})])

# a has shape (4,), b has shape (3,). Give each a new unlabeled axis so they
# broadcast into a (4, 3) result.
outer = a[:, None] * b[None, :]
assert outer.shape() == (4, 3)
```

The labeled axis keeps its labels, and the unlabeled axis is broadcast across it.

## Turning automatic alignment off

Automatic alignment can be disabled, either to avoid the cost of reordering or to catch
places where you expected arrays to already be aligned. The
[`config.auto_align`](../api/config.md) setting controls this. When it is off, binary
operations still *check* that their operands are aligned, and raise if they are not, but
they will not reorder anything.

```{.python continuation}
# With auto_align off, combining misaligned arrays raises instead of silently
# reordering them.
with pld.config.auto_align(False):
    try:
        result = array + shuffled
    except Exception as error:
        # Cannot combine arrays with unaligned labels.
        assert "unaligned labels" in str(error)
```

Called without an argument, `config.auto_align()` returns the current setting. Called with a
boolean it returns a context manager, so you can scope the change to a block as above. The
context managers nest, so an inner block can re-enable alignment within an outer block that
disabled it.

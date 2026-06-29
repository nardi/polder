# Pivoting

A single axis can carry several label columns. Sometimes those columns describe structure
that you would like to make explicit, by splitting the axis into several orthogonal axes.
This is what *pivoting* does, and *unpivoting* is its inverse.

This is an extension over the [array API](https://data-apis.org/array-api/latest/): instead
of reshaping by giving an explicit shape, you reshape by naming label columns, which keeps
the operation meaningful in terms of the labels.

## Pivoting an axis into several

Suppose an axis `i` is labeled by three columns, `x`, `y`, and `t`. If you know that you
have a value for each unique combination of `(x, y)` and `t`, you might want to make that
structure explicit, by splitting `i` into two axes, one labeled by `x` and `y`, and one
labeled by `t`. The `pivot` method does this.

```python
import numpy as np
import polars as pl

import polder as pld

values = np.arange(8 * 3).reshape(8, 3).astype(float)
labels = [
    pl.DataFrame({
        "x": [0, 0, 0, 0, 1, 1, 1, 1],
        "y": [0, 0, 1, 1, 0, 0, 1, 1],
        "t": [0, 1, 0, 1, 0, 1, 0, 1],
    }),
    pl.DataFrame({"extra": [0, 1, 2]}),
]
arr = pld.from_values_and_labels(values, labels)

# Pivot axis 0, moving "t" into a new axis.
pivoted = arr.pivot(axis_labels_to_pivot={0: ["t"]})
assert pivoted.shape() == (4, 2, 3)
```

The `axis_labels_to_pivot` mapping says which columns to pivot out of which axis. Here axis
0 had eight rows, made of four unique `(x, y)` pairs and two `t` values. Pivoting `t` into a
new axis turns the single axis of length 8 into one axis of length 4 (labeled by the
remaining `x` and `y` columns) followed by a new axis of length 2 (labeled by `t`). Any
columns you do not mention stay on the first axis.

## Splitting into more than two axes

You can split an axis into more than two new axes by giving a list of column groups. Each
group becomes its own axis, in order. The columns you do not mention stay on the first axis.

```{.python continuation}
# Split axis 0 into "x" (the remaining column), then "y", then "t".
split = arr.pivot(axis_labels_to_pivot={0: [["y"], ["t"]]})
assert split.shape() == (2, 2, 2, 3)
```

You can also be fully explicit and name every group, including the columns that stay. The
following produces the same result as the single-column form above, because the remaining
columns are listed as the first group.

```{.python continuation}
pivoted_explicit = arr.pivot(
    axis_labels_to_pivot={0: [["x", "y"], ["t"]], 1: ["extra"]}
)
assert pivoted_explicit.equals(pivoted)
```

Several axes can be pivoted in a single call by giving more than one entry in the mapping.

## Filling missing combinations

For a pivot to work, the array has to contain a value for each combination in the product of
the split labels. If some combinations are missing, the result would have holes, and you
have to say what to put in them with `fill_value`.

```{.python continuation}
sparse_labels = [
    pl.DataFrame({"x": [0, 0, 1, 1], "y": [0, 1, 0, 1], "t": [0, 0, 1, 1]}),
    pl.DataFrame({"extra": [0, 1]}),
]
sparse = pld.from_values_and_labels(np.arange(8.0).reshape(4, 2), sparse_labels)

# Not every (x, y, t) combination exists, so a fill value is required.
filled = sparse.pivot(axis_labels_to_pivot={0: ["t"]}, fill_value=np.nan)
assert filled.shape() == (4, 2, 2)
assert np.isnan(filled.values()).sum() == 8
```

If a fill value is needed but you do not provide one, pivoting raises an error. Because the
labels in the input are unique, no information is lost in any case, the only question is what
to put where there was no value.

```{.python continuation}
try:
    sparse.pivot(axis_labels_to_pivot={0: ["t"]})
except Exception as error:
    # ... no `fill_value` provided ...
    assert "fill_value" in str(error)
```

## Unpivoting

`unpivot` is the inverse of `pivot`. It merges adjacent axes back into one, by combining
their labels. You give the axes to merge as a list of `(start, end)` index pairs, where the
range is inclusive.

```{.python continuation}
# Undo the first pivot above, merging axes 0 and 1 back into one.
unpivoted = pivoted.unpivot(axes_to_merge=[(0, 1)])
assert unpivoted.shape() == (8, 3)
assert arr.equals(unpivoted)
```

A single merge can span more than two axes. For example `axes_to_merge=[(0, 2)]` merges
axes 0, 1, and 2 into one. You can perform several independent merges in one call by listing
more than one pair, as long as the pairs do not overlap, are in order, and each spans at
least two axes. Otherwise an error is raised.

```{.python continuation}
big = pld.from_values_and_labels(
    np.arange(4 * 2 * 2 * 2).reshape(4, 2, 2, 2).astype(float),
    [
        pl.DataFrame({"x": [0, 0, 1, 1], "y": [0, 1, 0, 1]}),
        pl.DataFrame({"t": [0, 1]}),
        pl.DataFrame({"s": [0, 1]}),
        pl.DataFrame({"extra": [0, 1]}),
    ],
)

# Merge axes (0, 1) and (2, 3) independently.
merged = big.unpivot(axes_to_merge=[(0, 1), (2, 3)])
assert merged.shape() == (8, 4)
```

# Development plan

This plan lists the work that is necessary to reach the goals in `README.md`. The eager
implementation comes first and must support the full array functionality. The lazy
implementation follows and must match the eager implementation. The last section lists
the operations that are hard to implement in the lazy implementation, each with a
suggested solution.

Statements about current behavior were checked against commit `2195794` with Narwhals
2.18.1 and Polars as the DataFrame backend.

## Goals from the README

| Id  | Goal                                                                                                                                     |
| --- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| G1  | The `FrameLabeledArray` protocol follows the array API and adds label operations such as `align` and `pivot`.                            |
| G2  | The eager implementation accepts each array that follows the array API. NumPy and JAX have tests and types.                              |
| G3  | The lazy implementation supports the full protocol and keeps all work in the DataFrame backend.                                          |
| G4  | An operation has the same cost profile as the backend operation. Automatic alignment is the only exception, and the user can disable it. |
| G5  | The user can convert an array between the implementations.                                                                               |
| G6  | A third implementation holds eager values and applies index operations lazily.                                                           |
| G7  | All library code is typed, and backend libraries other than `narwhals` and `numpy` are imported dynamically.                             |

## Current state

| Area                                                   | Eager               | Lazy                               |
| ------------------------------------------------------ | ------------------- | ---------------------------------- |
| `from_values_and_labels`                               | NumPy and JAX only  | Yes, values are converted to NumPy |
| `from_frame`                                           | No                  | Yes                                |
| `values`, `labels`, `shape`                            | Yes                 | Yes                                |
| `__getitem__`                                          | Yes, with gaps      | No                                 |
| `equals`                                               | Yes                 | No                                 |
| `pivot`, `unpivot`                                     | Yes                 | Yes                                |
| `align`                                                | Yes                 | Yes, `check_only` has no effect    |
| Operators for arithmetic, bitwise logic and comparison | Yes                 | Yes                                |
| `__matmul__`                                           | Two dimensions only | No                                 |
| Unary functions                                        | 39                  | 22                                 |
| All other array API functions and attributes           | No                  | No                                 |

The following defects were found in the existing code.

1. The eager binary operators convert a Python scalar to an array with `xp.full`. An
   `int8` array plus `1` returns `int64`, while NumPy returns `int8`.
1. The eager `__getitem__` raises an error for `...` and for a boolean mask.
1. The eager `__matmul__` raises an error for stacked matrices, and `__rmatmul__` does
   not exist.
1. The binary operators raise `NotImplementedError` for an unknown operand type. Python
   expects the value `NotImplemented`, so that the other operand can handle the
   operation.
1. `from_values_and_labels` with `implementation=EAGER` raises `NotImplementedError` for
   an array that is not a NumPy or JAX array.
1. The eager `pivot` calls `xp.take` with a two-dimensional index array. The array API
   only defines `take` for a one-dimensional index array.
1. The lazy implementation ignores `config.auto_align`. Two lazy arrays with labels in a
   different order are aligned when the setting is `False`.
1. A lazy binary operation on arrays with different labels returns an array whose
   `labels` call succeeds. Only `values` raises an error, and the message does not name
   the cause.
1. `LazyFrameLabeledArray.from_values_and_labels` raises an error for a zero-dimensional
   array.

## Design decisions

These decisions apply to both implementations, so they come before the implementation
work.

### Operation dispatch

The functions in `polder.operations` select an implementation with `isinstance` chains.
A registry per operation, for example with `functools.singledispatch`, lets a new
implementation register its own function without a change in `polder.operations`. The
protocol then only contains the special methods, the attributes, and the label
operations. All other array API functions are free functions in the `polder` namespace,
as in the array API itself.

### Label rules

Each array API function needs a rule for the labels of its result. The table gives the
proposed rule per function category.

| Category         | Functions                                                                        | Labels of the result                                                                       |
| ---------------- | -------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| Elementwise      | Unary and binary functions, `clip`, `where`                                      | Unchanged after alignment                                                                  |
| Reduction        | `sum`, `prod`, `mean`, `min`, `max`, `std`, `var`, `all`, `any`, `count_nonzero` | The reduced axis is removed. With `keepdims=True`, it becomes an unlabeled axis of size 1. |
| Index reduction  | `argmin`, `argmax`                                                               | As a reduction. The values are integer positions.                                          |
| Cumulative       | `cumulative_sum`, `cumulative_prod`                                              | Unchanged. `include_initial=True` adds an entry without a label, so it is not supported.   |
| Axis order       | `permute_dims`, `moveaxis`, `matrix_transpose`, `T`, `mT`                        | The label frames are permuted with the axes.                                               |
| Order in an axis | `flip`, `roll`, `take`                                                           | The label frame is reordered with the values.                                              |
| Size-1 axes      | `expand_dims`, `squeeze`                                                         | An unlabeled axis is added or removed. `squeeze` of a labeled axis is an error.            |
| Combination      | `concat`                                                                         | The label frames of the axis are concatenated. All other axes are aligned.                 |
| Combination      | `stack`                                                                          | A new axis is added. Its labels come from a new `labels` argument.                         |
| Contraction      | `matmul`, `tensordot`, `vecdot`                                                  | The contracted axes are aligned and removed. The other label frames are kept in order.     |
| Shape change     | `diff`                                                                           | See [Shape changes](#shape-changes).                                                       |
| Sort             | `sort`, `argsort`                                                                | See [Sort functions](#sort-functions).                                                     |
| Shape from data  | Boolean mask index, `nonzero`, `unique_*`                                        | See [Shapes from data](#shapes-from-data).                                                 |
| Creation         | `zeros_like`, `ones_like`, `full_like`, `empty_like`                             | The labels of the input array                                                              |
| Conversion       | `astype`, `__bool__`, `__int__`, `__float__`, `__complex__`, `__index__`         | Unchanged. The scalar conversions need a zero-dimensional array.                           |

### Shape changes

`diff` along an axis of size `n` returns `n - 1` entries, so one label row has no entry.
The proposed rule keeps the label of the later element of each pair, which is also the
row that a DataFrame `diff` fills. An argument `label_side="lower"` selects the label of
the earlier element. For `n > 1` orders of difference, the rule is applied `n` times.
The `prepend` and `append` arguments must be frame-labeled arrays, and they are
concatenated with `concat` before the difference.

### Sort functions

`sort` along an axis of a one-dimensional array reorders the label frame with the
values. In an array with more dimensions, each lane has its own order, so the labels of
the sorted axis have no meaning afterwards. The proposed rule replaces the label frame
of that axis with a frame that has one integer column, named by a `rank_column` argument
with the default `"rank"`. `argsort` uses the same rule for its labels.

### Shapes from data

1. A boolean mask with one dimension filters one axis and its label frame.
1. A boolean mask with the full shape of the array first applies `unpivot` to all axes,
   and then filters the merged axis. The result has one dimension, and its label frame
   contains the columns of all axes.
1. `nonzero` returns integer arrays in the array API. A polder function `nonzero_labels`
   can return the label rows of the nonzero entries as one frame, which is the result
   that a user of labels needs.
1. `unique_values` returns a bare value array, because the unique values have no labels.
   `unique_counts`, `unique_inverse` and `unique_all` are left out until a use case
   exists.

Under `jax.jit`, the shape of these results is not known at trace time. These functions
therefore only work on eager JAX arrays outside of a traced function.

### Functions without labels

The following functions get no implementation, and the documentation must state the
alternative.

| Function                                                 | Reason                                                                           | Alternative                                                                 |
| -------------------------------------------------------- | -------------------------------------------------------------------------------- | --------------------------------------------------------------------------- |
| `reshape`                                                | A general reshape has no rule for the labels.                                    | `pivot`, `unpivot`, `expand_dims`, `squeeze`                                |
| `repeat`, `tile`                                         | The result contains each label more than one time, so alignment is not possible. | `stack` or `concat` with new labels                                         |
| `broadcast_to`                                           | The new entries of the axis need labels.                                         | A binary operation with an array that has the labels, or `broadcast_arrays` |
| `arange`, `linspace`, `eye`, `meshgrid`, `zeros`, `ones` | The result has no labels.                                                        | `from_values_and_labels`, the `*_like` functions                            |
| `__setitem__` and the in-place operators                 | All arrays are immutable.                                                        | `where`                                                                     |
| Complex numbers in the lazy implementation               | The DataFrame backends have no complex data type.                                | Conversion to the eager implementation                                      |

The array API allows a binary operation on arrays with a different number of dimensions.
polder keeps the current error, because an axis that is added without a statement from
the user can pair the wrong label frames. The user adds the axis with `None` in an
index.

### Unique labels

Alignment and `pivot` assume that the rows of a label frame are unique, but no code
checks this. `concat` and an integer array index with a repeated position can create a
repeated row. The proposal is a check in `align` and `pivot` that raises an error for a
repeated row, and no check at creation, so that creation has no extra cost.

## Eager implementation

Each stage is one or a few pull requests. Each new function gets a test against the same
NumPy function on the bare values, for NumPy and JAX arrays.

### Stage 1: defects and test backends

1. Correct defects 1 to 6 from the list above.
1. Add `array-api-strict` to the `dev` dependency group and to the `value_array_type`
   fixture. This package only accepts calls that the array API defines, so it finds each
   use of a NumPy extension.
1. Accept each array for which `array_api_compat.is_array_api_obj` returns `True` in
   `from_values_and_labels`.
1. Add `from_frame` for the eager implementation, with an argument for the array
   namespace of the values.
1. Change the dispatch in `polder.operations` to a registry.

### Stage 2: attributes and conversions

1. Add `dtype`, `ndim`, `size`, `device`, `T` and `mT`.
1. Add `astype`, `to_device`, and the scalar conversions.
1. Add `zeros_like`, `ones_like`, `full_like` and `empty_like`.
1. Decide if `shape` stays a method. The array API defines it as an attribute, and a
   later change breaks user code.

### Stage 3: elementwise functions

1. Add the function form of each binary operator, such as `add`, `multiply`, `equal` and
   `bitwise_and`.
1. Add the binary functions without an operator: `atan2`, `hypot`, `copysign`,
   `logaddexp`, `maximum`, `minimum`, `nextafter`, `logical_and`, `logical_or` and
   `logical_xor`.
1. Add `clip` and `where`. These take up to three arrays, so `align` must run on all of
   them.
1. Add `map_values(array, function)`, which applies a function that keeps the shape to
   the value array. It gives access to backend functions that polder does not wrap, such
   as `jax.scipy.special` functions.

### Stage 4: reductions

1. Add the reductions and index reductions from the label rules table, with `axis` and
   `keepdims`.
1. Add `cumulative_sum` and `cumulative_prod`.
1. Allow `axis=None`, which returns a zero-dimensional array.

### Stage 5: axis manipulation

1. Add `permute_dims`, `moveaxis`, `matrix_transpose`, `expand_dims` and `squeeze`.
1. Add `flip`, `roll`, `take` and `take_along_axis`.
1. Add `concat`, `stack`, `unstack` and `broadcast_arrays`.
1. Add the check for unique labels.

### Stage 6: shape changes

1. Add `diff`, `sort` and `argsort` with the label rules above.
1. Add the boolean mask index, `nonzero_labels` and `unique_values`.
1. Add `searchsorted`, which needs a one-dimensional sorted array and returns integer
   positions.

### Stage 7: linear algebra

1. Extend `matmul` to stacked matrices and one-dimensional operands. The stack axes are
   aligned as in a binary operation.
1. Add `tensordot` and `vecdot`.
1. Leave the `linalg` and `fft` extensions of the array API out. `map_values` covers the
   functions that keep the shape.

### Stage 8: array namespace

When stages 2 to 7 are complete, add `__array_namespace__` so that `polder` is the array
namespace of its arrays. The documentation must list each deviation from the array API,
which are the functions in [Functions without labels](#functions-without-labels) and the
extra arguments for labels. The `array-api-tests` suite cannot run without changes,
because it creates arrays without labels.

## Lazy implementation

Each function has one shared test in `tests/operations` that runs on both
implementations. The test converts the lazy result with `convert` and compares it to the
eager result with `equals`, so the eager implementation is the reference. The tests run
with `use_eager_evaluation_for_lazy_arrays` on and off.

### Stage 9: internal structure

1. Add the deferred checks from [Deferred validation](#deferred-validation), and use
   them for defects 7 and 8.
1. Support zero-dimensional arrays, which is defect 9. The value frame then has one row
   and no index column.
1. Add `dtype` from `collect_schema`, and cast the result of each operation to the data
   type that the array API prescribes. See [Data types](#data-types).
1. Skip the joins in `align` when the label frames of both arrays are the same objects.
   This is the common case after an elementwise operation on one array.
1. Apply the index of `values[...]` as a filter before `collect`.

### Stage 10: protocol parity

1. Add `__getitem__`. See [Index operations](#index-operations).
1. Add `equals`. It returns a `bool`, so it must collect, as `values` does.
1. Add `__matmul__` as a join on the index column of the contracted axis, followed by
   `group_by` and `sum`.
1. Add the missing unary functions. See [Missing expressions](#missing-expressions).

### Stage 11: function parity

The stages 2 to 7 are repeated for the lazy implementation, in the same order. Most
functions follow one of four patterns.

| Pattern                 | Functions                                                            | Lazy plan                                                                                           |
| ----------------------- | -------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| Expression on `value`   | Elementwise functions, `astype`, `clip`, `where`                     | `with_columns` on the value frame, after a join for more than one array                             |
| Change of index columns | `permute_dims`, `moveaxis`, `expand_dims`, `squeeze`, `flip`, `roll` | `rename` or an expression on the `__index` columns and the shape frame                              |
| Aggregation             | Reductions, `matmul`, `tensordot`, `vecdot`                          | `group_by` on the index columns that stay, then an aggregation                                      |
| Window                  | `cumulative_sum`, `cumulative_prod`, `diff`, `sort`, `argsort`       | An expression with `over` on the other index columns and `order_by` on the index column of the axis |

### Stage 12: backends and the third implementation

1. Run the lazy tests on at least one more Narwhals backend with lazy frames, such as
   DuckDB. Only Polars is tested at this time.
1. Design the third implementation from the README. It can reuse the eager value array,
   and the label frames and index mappings of the lazy implementation. A pending index
   mapping is applied to the values with one `take` per axis when `values` is called.
1. Extend `convert` with the third implementation.

## Lazy implementation difficulties

### Deferred validation

A lazy operation cannot raise an error for a condition that depends on data, such as
labels that do not match. The shape frame currently has this role: an invalid operation
creates a missing or repeated row, and `shape` then raises "Array has invalid shape".
This message does not name the operation, and some conditions have no effect on the
shape frame.

The suggested solution is a new field `_checks`, which is a lazy frame with the columns
`message` and `passed`. Each operation that can fail appends one row, with an expression
that computes `passed`. `values`, `shape`, `labels` and `collect` evaluate the frame and
raise the first message that did not pass. The conditions that need a check are:

1. The labels of two arrays contain the same rows. This is the alignment check.
1. The labels of two arrays are in the same order, when `config.auto_align` is `False`.
1. An index with a mapping, such as `{"year": 2021}`, matches exactly one label row.
1. An integer index is within the size of its axis.
1. The label rows are unique, for `align` and `pivot`.

### Index operations

1. An integer, a slice or an integer array with a negative position needs the size of
   the axis. A call to `shape()` collects the shape frame. A cross join with the one-row
   size of the axis keeps the plan lazy, and it was tested for `flip`.
1. An index that removes or reorders entries must renumber the `__index` column to the
   range from 0 to the new size. The lazy `pivot` already does this with
   `with_row_index(order_by=...)` on the filtered label frame. The value frame then gets
   the new index with an inner join on the old index.
1. An integer array index can be converted to a mapping frame from old to new position
   before the plan is built, because the index array is eager data.
1. An expression index and a one-dimensional boolean mask filter the label frame. This
   is simpler than in the eager JAX case, because the long format has no fixed shape.
1. The number of dimensions after an index is known without a collect, because it only
   depends on the index types and the label columns.

### Operations that change the shape

`diff` along axis `k` was tested as
`nw.col("value").diff().over(other_index_columns, order_by=index_column)`. The plan then
removes the rows where the index of axis `k` is 0, subtracts 1 from that index in the
value frame and the label frame, and subtracts 1 from the size in the shape frame. The
new shape follows from the old shape with a fixed rule, so no data is needed. The same
holds for `concat`, where the offset of each array is the sum of the sizes before it,
and for `stack`, `expand_dims` and `squeeze`.

A shape that depends on data, as for a boolean mask or `unique_values`, is computed as a
count of the new label frame. This is the method of the lazy `pivot`.

### Order in a lazy frame

A Narwhals `LazyFrame` has no row order. `with_row_index` requires `order_by`, `unique`
has no `maintain_order` argument, and an order-dependent expression such as
`map_batches` raises an error. Each window operation must therefore name its order with
`over(..., order_by=...)`.

`sort` along an axis needs a rank per lane with a fixed order for equal values.
`rank("ordinal")` has no second sort key, so equal values get an order that is not
defined. The tested solution is
`with_row_index("r", order_by=[*other_index_columns, "value", index_column])`. The new
index is then `r` minus the minimum of `r` over the other index columns.

### Missing expressions

Narwhals has no expression for `tan`, the inverse trigonometric functions, the
hyperbolic functions, `log1p`, `sign`, `trunc`, `isinf`, `atan2`, `hypot`, `copysign`,
`nextafter`, and `prod` as an aggregation. Bitwise `xor` and the shifts are also
missing. The current code replaces a shift with a multiplication or division by a power
of 2, which has a different overflow behavior. There are three solutions, in order of
preference.

1. Derive the function from existing expressions when the result has the same accuracy.
   Examples are `tan` as `sin / cos`, `sign` and `trunc` with `nw.when`, and `isinf` as
   not finite and not NaN.
1. Add a table of native expressions per backend. Polars has `arctan`, `tanh` and most
   of the other missing functions. The operation then converts the frame with
   `to_native`, applies the native expression, and wraps the result with `from_native`.
   A backend without an entry raises `NotImplementedError`.
1. Propose the missing expressions to Narwhals, and remove the table entries when a
   release contains them.

`log1p` and `expm1` are a special case. The derived forms `log(1 + x)` and `exp(x) - 1`
lose accuracy for small values, so the existing lazy `expm1` does not have the accuracy
that the array API function implies. They need solution 2. `nextafter` has no DataFrame
equivalent and stays unsupported.

A call to a Python function on the values inside the plan is not possible, as
`map_batches` is not allowed on a lazy frame. `map_values` therefore has no lazy
implementation, and the documentation must refer to `convert`.

### Data types

The array API prescribes the data type of each result, and the DataFrame backends follow
other rules. The following differences were measured with Polars.

| Expression            | NumPy | Polars through Narwhals |
| --------------------- | ----- | ----------------------- |
| `max` of `[1.0, nan]` | `nan` | `1.0`                   |
| `sum` of `[1.0, nan]` | `nan` | `nan`                   |

Floor division and remainder of `-7` and `2`, and the overflow of `int8` addition, gave
the same results as NumPy. Other backends can differ, in particular SQL backends for
integer division. The suggested solution has three parts.

1. Compute the result data type with the array API promotion rules from the data types
   in `collect_schema`, and cast the `value` column after each operation.
1. Make NaN propagation explicit in `min`, `max`, `argmin` and `argmax`, with an
   `is_nan().any()` aggregation and `nw.when`.
1. Keep null and NaN separate. A null in the `value` column always means that an entry
   is missing, and `values` raises an error for it. An operation that creates nulls,
   such as `diff`, must remove those rows in the same step.

### Reductions

A reduction is a `group_by` on the index columns that stay. Three cases need extra work.

1. A reduction over an axis of size 0 returns no rows, while the array API prescribes
   the identity value, such as 0 for `sum`. The plan needs a left join from the full
   index of the result and a `fill_null` with the identity value.
1. `argmin` and `argmax` have no aggregation in Narwhals. The plan can filter the rows
   where the value equals the minimum over the lane, and then take the minimum index.
1. `prod` has no aggregation in Narwhals. It needs the native expression table, because
   `exp(sum(log(x)))` loses accuracy and does not handle zero or negative values.

### Plan size

Each binary operation adds one join per labeled axis for alignment, and one join for the
values. A long chain of operations creates a deep plan, and a hash join on all index
columns costs more than the same elementwise operation on an array. This conflicts with
G4 for dense numeric code. Three measures limit the cost.

1. Skip alignment for label frames that are the same objects, as in stage 9.
1. Document `collect` as the method that ends the current plan and stores its result.
1. State in the user guide that the lazy implementation fits relational work, and that
   `convert` to the eager implementation fits dense numeric loops.

### Methods that return Python values

`equals`, `shape`, `__bool__`, `__int__`, `__float__` and `__index__` return Python
values, so each call collects the plan. This matches the statement in the README that a
computation is resolved when its values are converted to an eager form. The
documentation of each method must state that the call collects.

### Backend coverage

`_create_index_df` and the creation functions build frames with `nw.from_numpy` in the
native namespace of the label frames. This was only run with Polars. A backend that only
has lazy frames, such as DuckDB or PySpark, can need another creation path, and stage 12
must test this before the README can claim support for each Narwhals backend.

## Open questions

1. Should `shape` become an attribute to match the array API?
1. Should the reductions accept label columns in addition to an axis number, such as
   `sum(array, over={0: ["year"]})`? This is equal to a `pivot` followed by a reduction,
   and it avoids the dense intermediate array.
1. Should the check for unique labels run at creation, with a configuration setting to
   disable it, instead of in `align` and `pivot`?
1. Is `rank` the correct default label for a sorted axis, or should `sort` on an array
   with more than one dimension be an error?

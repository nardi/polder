# Development plan

This plan lists the work that is necessary to reach the goals in `README.md`. The eager
implementation comes first and must support the full array functionality. The lazy
implementation follows and must match the eager implementation.

The plan has five parts.

1. [Current state](#current-state) lists what exists, the defects that were found, and
   the measurements that the decisions refer to.
2. [Design decisions](#design-decisions) contains the choices that apply to both
   implementations. Each decision has its reason and the alternatives that were
   rejected.
3. [Eager implementation](#eager-implementation) and
   [Lazy implementation](#lazy-implementation) contain the stages, with the
   implementation details per stage.
4. [Lazy implementation difficulties](#lazy-implementation-difficulties) lists the
   operations that are hard to implement lazily, each with a suggested solution.
5. [Open questions](#open-questions) lists the choices that need a decision from the
   maintainer.

Statements about current behavior were checked with an experiment against commit
`2195794`, with Narwhals 2.18.1 and Polars. A statement about pandas, PyArrow, DuckDB or
`array-api-strict` was checked with that package installed next to the `dev` dependency
group. A code block in this plan is a sketch that shows a structure. It is not final
code.

## Goals from the README

| Id  | Goal                                                                                                                                                           |
| --- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| G1  | The `FrameLabeledArray` protocol mirrors the array API and adds label operations such as `align` and `pivot`. It is not a conforming array API implementation. |
| G2  | The eager implementation accepts each array that follows the array API. NumPy and JAX have tests and types.                                                    |
| G3  | The lazy implementation supports the full protocol and keeps all work in the DataFrame backend.                                                                |
| G4  | An operation has the same cost profile as the backend operation. Automatic alignment is the only exception, and the user can disable it.                       |
| G5  | The user can convert an array between the implementations.                                                                                                     |
| G6  | A third implementation holds eager values and applies index operations lazily.                                                                                 |
| G7  | All library code is typed, and backend libraries other than `narwhals` and `numpy` are imported dynamically.                                                   |

## Current state

### Feature coverage

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
| All other array API functions and accessors            | No                  | No                                 |
| Label frames from Polars and pandas                    | Yes                 | Yes                                |
| Label frames from PyArrow                              | Yes                 | Creation only                      |

### Defects

Each defect has an identifier that the stages refer to.

Defects in the eager implementation:

- **D1.** The binary operators convert a Python scalar to an array with `xp.full`. An
  `int8` array plus `1` returns `int64`, while NumPy returns `int8`.
- **D2.** `__getitem__` raises an error for `...` and for a boolean mask.
- **D3.** `__matmul__` raises an error for stacked matrices, and `__rmatmul__` does not
  exist.
- **D4.** The binary operators raise `NotImplementedError` for an unknown operand type.
  Python expects the return value `NotImplemented`, so that the other operand can handle
  the operation.
- **D5.** `from_values_and_labels` with `implementation=EAGER` raises
  `NotImplementedError` for an array that is not a NumPy or JAX array.
- **D6.** The index code uses NumPy extensions of the array API. With `array-api-strict`
  arrays, the binary operators, `__getitem__`, `align` and `__matmul__` raise
  `IndexError`, because an index tuple is shorter than the number of dimensions. `pivot`
  raises `ValueError`, because it calls `xp.take` with a two-dimensional index array.
  Unary functions, `unpivot` and `equals` work.
- **D7.** `align` joins the label frames and copies the values of each operand after the
  first, also when the labels are already equal. See [Measurements](#measurements).
- **D8.** A check with `auto_align(False)` costs more than an alignment, so the setting
  does not meet G4.
- **D9.** Most errors are raised as a bare `Exception`, so user code cannot catch an
  alignment error separately.
- **D10.** `align` is not exported from the `polder` package, while the README lists it
  as a generic operation.

Defects in the lazy implementation:

- **D11.** `config.auto_align` is ignored. Two lazy arrays with labels in a different
  order are aligned when the setting is `False`.
- **D12.** A binary operation on arrays with different labels returns an array whose
  `labels` call succeeds. Only `values` raises an error, with the message "Array has
  indices for which no value is stored".
- **D13.** `from_values_and_labels` raises an error for a zero-dimensional array.
- **D14.** With PyArrow label frames, each binary operation, `pivot` and `unpivot`
  raises `ArrowInvalid`. The index columns from `_create_index_df` are `uint32`, and the
  index columns from `with_row_index` are `int64`. PyArrow does not join columns of
  different types.
- **D15.** An `int8` array plus `1` returns `int64`, and a `bool` array plus a `bool`
  array returns `uint32`. The eager implementation returns `int8` after the correction
  of D1.
- **D16.** The lazy frames always use the backend of the label frames. A lazy engine
  such as DuckDB cannot be selected, because `nw.from_numpy` does not accept a backend
  that only has lazy frames.

### Measurements

The numbers are the mean of a few runs on one machine, so only the ratios are
meaningful.

Binary operation `x + y` on one-dimensional arrays with one integer label column:

| Case                                  | 1 000 entries | 1 000 000 entries |
| ------------------------------------- | ------------- | ----------------- |
| NumPy on the bare values              | 0.002 ms      | 1.0 ms            |
| Eager, `auto_align` on, labels equal  | 0.9 ms        | 43 ms             |
| Eager, `auto_align` off, labels equal | 1.5 ms        | 138 ms            |
| Eager, `x + x`                        | 0.8 ms        | 40 ms             |
| Lazy, plan and `values()`             | 4.7 ms        | 121 ms            |
| Lazy, `values()` without an operation | 1.1 ms        | 45 ms             |

Label equality check on two frames with 1 000 000 rows and two columns:

| Method                                                     | Time   |
| ---------------------------------------------------------- | ------ |
| `narwhals_df_equals`, which uses a full join               | 133 ms |
| Comparison per column with `Series.__eq__` and a null mask | 5.6 ms |

Chain of `k` additions on a 300 by 300 array:

| `k` | Eager  | Lazy   |
| --- | ------ | ------ |
| 1   | 1.5 ms | 20 ms  |
| 4   | 6.8 ms | 41 ms  |
| 8   | 69 ms  | 603 ms |

Three conclusions follow from these numbers.

1. In the eager implementation, the label work costs about 40 times more than the array
   operation, also for `x + x`. This is the largest gap to G4.
2. The setting that should remove the alignment cost makes the operation three times
   slower.
3. The lazy implementation has a fixed cost of about 45 ms per million entries for
   `values()`, because it joins the values onto a dense index and sorts them.

## Design decisions

### Array API relation

polder mirrors the array API and does not conform to it. A function keeps the name, the
arguments and the value semantics of the array API function when a label rule exists for
it. The protocol deviates in three places.

1. A property of the array that can start a computation is a method, as `shape()` is. In
   the lazy implementation, such a call can collect a frame. A property that never
   starts a computation, such as `T`, can be a Python property.
2. A function has extra arguments when the labels of the result need them.
3. A function without a label rule is left out. See
   [Functions without labels](#functions-without-labels).

As a consequence, `polder` is not an array namespace, and `__array_namespace__` is not
planned. A library that accepts array API arrays does not accept a frame-labeled array.
The user passes `array.values()` to such a library and adds the labels to the result.

### Function location

The protocol contains the special methods, the accessor methods, `equals`, `pivot` and
`unpivot`. All other functions are free functions in `polder.operations`, which the
`polder` package exports again.

The reason is the statement in the README that a change of the protocol is significant.
A third-party implementation of the protocol must then only provide the methods, and it
can register the free functions one at a time. The array API has the same division
between array methods and namespace functions, so user code that mirrors array API code
has the same form.

The rejected alternative is a method for each function, such as `array.sum(axis=0)`. It
makes the protocol about 100 members larger, and each new function breaks each
implementation outside of the package.

### Operation dispatch

The functions in `polder.operations` select an implementation with `isinstance` chains.
Each function becomes a `functools.singledispatch` function. An implementation registers
its own function in its own subpackage. This was tested with `EagerFrameLabeledArray`.

```python
# polder/operations/reductions.py
@singledispatch
def sum(array: FrameLabeledArray, /, *, axis: AxisArgument = None, keepdims: bool = False):
    raise NotImplementedError(f"`sum` is not implemented for {type(array).__name__}.")

# polder/eager/reductions.py
@operations.sum.register
def _(array: EagerFrameLabeledArray, /, *, axis=None, keepdims=False): ...
```

A function with more than one array argument dispatches on the first array. The eager
and lazy functions raise `TypeError` when the other arrays have another implementation.
A mix of implementations is not converted automatically, because a conversion from lazy
to eager collects the plan, and that cost must be visible in user code.

The registration must not create an import cycle. `polder/__init__.py` imports
`polder.operations` first and the implementation subpackages second, and only the
implementation subpackages import `polder.operations`.

### Function names

The existing unary functions use a trailing underscore for a name that is also a Python
builtin, as in `abs_` and `round_`. The new functions `sum`, `min`, `max`, `all`, `any`
and `pow` have the same conflict. The plan keeps the existing rule and exports `sum_`,
`min_`, `max_`, `all_`, `any_` and `pow_`, so that one rule applies to the full
namespace. This is [open question 1](#open-questions), because `pld.sum` is the name
that a NumPy user expects.

### Axis arguments

An `axis` argument accepts an integer, and a negative integer counts from the last axis.
One helper `normalize_axis(axis, ndim)` in `polder/utils/axis.py` converts the argument
and raises `IndexError` for a value outside of the range. Both implementations use it,
because the number of dimensions is known without a computation in both.

A function that reduces more than one axis accepts a tuple of integers. `axis=None`
selects all axes.

### Label rules

Each function needs a rule for the labels of its result. The table gives the proposed
rule per function category.

| Category         | Functions                                                                        | Labels of the result                                                                       |
| ---------------- | -------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| Elementwise      | Unary and binary functions, `clip`, `where`                                      | Unchanged after alignment                                                                  |
| Reduction        | `sum`, `prod`, `mean`, `min`, `max`, `std`, `var`, `all`, `any`, `count_nonzero` | The reduced axis is removed. With `keepdims=True`, it becomes an unlabeled axis of size 1. |
| Index reduction  | `argmin`, `argmax`                                                               | As a reduction. The values are integer positions.                                          |
| Cumulative       | `cumulative_sum`, `cumulative_prod`                                              | Unchanged. `include_initial=True` adds an entry without a label, so it is not supported.   |
| Axis order       | `permute_dims`, `moveaxis`, `matrix_transpose`, `T`                              | The label frames are permuted with the axes.                                               |
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

The general principle is that a label row stays with its entries. When the entries of an
axis are combined into one entry, the label has no meaning and the axis becomes
unlabeled or is removed. This principle gives the rules for the reductions and for
`keepdims`. An unlabeled axis of size 1 is also the axis that a binary operation
broadcasts, so `array - mean(array, axis=0, keepdims=True)` works without extra code.

`argmin` and `argmax` return positions, as in the array API. A position refers to the
label order of the input array, so it is only valid for that array. A function that
returns the label row of the extreme entry is [open question 2](#open-questions).

### Shape changes

`diff` along an axis of size `n` returns `n - 1` entries, so one label row has no entry.
The proposed rule keeps the label of the later element of each pair. An argument
`label_side="lower"` selects the label of the earlier element.

The reason for the default is that the most common axis for `diff` is a time axis, and
the difference at time `t` is then the change since the previous time. A DataFrame
`diff` fills the same row. The rejected alternative is a result with an unlabeled axis,
which loses the labels of all entries for a problem with one row.

For `n > 1` orders of difference, the rule is applied `n` times. The `prepend` and
`append` arguments must be frame-labeled arrays, so that their entries have labels, and
they are concatenated with `concat` before the difference.

### Sort functions

`sort` along an axis of a one-dimensional array reorders the label frame with the
values. In an array with more dimensions, each lane has its own order, so the labels of
the sorted axis have no meaning afterwards. The proposed rule replaces the label frame
of that axis with a frame that has one integer column, named by a `rank_column` argument
with the default `"rank"`. `argsort` uses the same rule for its labels.

The rank column makes the result a normal labeled array, so `sorted_array[{"rank": 0}]`
returns the minimum per lane with the labels of all other axes. The rejected alternative
is an unlabeled axis, which is only allowed for size 1 in the current data model.

### Shapes from data

1. A boolean mask with one dimension filters one axis and its label frame. The mask is a
   frame-labeled array, and it is aligned with that axis first.
2. A boolean mask with the full shape of the array first applies `unpivot` to all axes,
   and then filters the merged axis. The result has one dimension, and its label frame
   contains the columns of all axes. This is the labeled form of the NumPy result, which
   is also one-dimensional.
3. `nonzero` returns integer arrays in the array API. A polder function `nonzero_labels`
   returns the label rows of the nonzero entries as one frame, which is the result that
   a user of labels needs.
4. `unique_values` returns a bare value array, because the unique values have no labels.
   `unique_counts`, `unique_inverse` and `unique_all` are left out until a use case
   exists.

Under `jax.jit`, the shape of these results is not known at trace time. These functions
therefore only work on eager JAX arrays outside of a traced function. The other eager
operations work under `jax.jit` and `jax.grad`, which was tested for a binary operation
and for an index with a mapping.

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

### Label identity

Two arrays that come from the same source often contain the same label frame object.
`sin(array)` already returns an array with the label objects of its input. A binary
operation does not, because `_align_labels` returns `label_frames[0].select(columns)`,
which is a new object.

The decision is that each operation returns the label frame objects of its input when
the labels do not change, and that `align` compares objects with `is` before it compares
rows. This gives three levels of alignment cost.

1. Same object: no label work and no copy of the values.
2. Equal rows: one comparison per column, which took 5.6 ms for a million rows, and no
   copy of the values.
3. Different order: a join and a reorder of the values, as today.

The rejected alternative is a cache of comparison results, with the identity of the
frames as the key. It needs weak references to objects that Narwhals creates again on
each `from_native` call, so the cache has few hits.

### Unique labels

Alignment and `pivot` assume that the rows of a label frame are unique, but no code
checks this. `concat` and an integer array index with a repeated position can create a
repeated row. The proposal is a check in the join path of `align` and in `pivot` that
raises an error for a repeated row. There is no check at creation, so that creation has
no extra cost, and levels 1 and 2 of the alignment also have no check.

### Scalar operands

A Python scalar is passed to the backend operator without a conversion. The backend then
applies its own promotion rule, in which the scalar takes the data type of the array.
This corrects D1 and removes the temporary array. The result has the labels of the array
operand.

### Error types

A new module `polder/exceptions.py` defines the errors, so that user code can catch
them.

| Class                 | Base class   | Raised when                                                                               |
| --------------------- | ------------ | ----------------------------------------------------------------------------------------- |
| `LabelAlignmentError` | `ValueError` | Label frames have different columns or rows, or are not aligned while `auto_align` is off |
| `DuplicateLabelError` | `ValueError` | A label frame has a repeated row in a join                                                |
| `UnlabeledAxisError`  | `ValueError` | An operation needs labels on an unlabeled axis                                            |

Each class has a standard base class, so existing code that catches `ValueError` works.
The bare `Exception` and `KeyError` in `eager/align.py`, `lazy/align.py` and
`eager/array.py` are replaced.

## Eager implementation

Each stage is one or a few pull requests. Each stage also updates the user guide and the
table in `docs/api/operations.md`.

### Eager test method

Each new function gets a test against the same NumPy function on the bare values. The
`value_array_type` fixture in `tests/conftest.py` runs the test for NumPy, JAX and
`array-api-strict`. The test also compares the labels of the result with an expected
tuple of frames.

Each function with a label rule gets a test with shuffled labels from `shuffle_labels`,
to show that the rule holds after alignment. The functions that keep the shape get one
test under `jax.jit`.

### Stage 1: defects and backend independence

This stage contains no new functions. It comes first because each later stage builds on
the index helper, the alignment and the dispatch.

**Index helper.** A new function `index_axis(xp, values, axis, index)` in
`polder/eager/value_array.py` replaces each expression of the form
`values[(slice(None),) * axis + (index,)]`. It corrects D6.

```python
def index_axis(xp, values, axis, index):
    """Index `values` along one axis and keep all other axes."""
    if is_integer_array(index):
        # `take` is the array API function for an integer array index.
        return xp.take(values, index, axis=axis)
    # A trailing `...` makes the index valid for each number of dimensions.
    return values[(slice(None),) * axis + (index, ...)]
```

`pivot` flattens its two-dimensional index array, calls `xp.take`, and reshapes the
result with `xp.reshape`.

**`__getitem__`.** The loop over the indices first expands `...` to the number of
`slice(None)` entries that fills the remaining axes. A boolean mask is converted to
integer positions with `xp.nonzero` and then handled as an integer array. The array API
only defines a boolean mask as the single index, so the conversion is necessary for a
mask on one axis. This corrects D2.

**Alignment.** `align` gets the three cost levels from
[Label identity](#label-identity). This corrects D7 and D8.

```python
def labels_are_equal(left, right) -> bool:
    """Compare two label frames per column, with null equal to null."""
    if left is right:
        return True
    if left.columns != right.columns or len(left) != len(right):
        return False
    return all(
        ((left[c] == right[c]) | (left[c].is_null() & right[c].is_null())).all()
        for c in left.columns
    )
```

`_align_labels` calls `labels_are_equal` for each frame against the first frame. When
all frames are equal, it returns the original frame objects and `slice(None)` for each
array, and `align` does not index the values. Only a frame that is not equal goes
through the join. `_check_label_alignment` uses the same function, so a check never
costs more than an alignment. `narwhals_df_equals` is removed, and `Labels.__eq__` and
`equals` use `labels_are_equal`.

Columns in a different order are equal labels in the join path today. `labels_are_equal`
therefore compares after a `select` in the column order of the first frame when the
column sets are equal.

**Binary operators.** `_perform_binop` passes a scalar to the operator directly and
returns `NotImplemented` for an unknown operand type. This corrects D1 and D4.

**Matrix product.** `matmul` follows the array API rules for the shapes. The last axis
of the left operand is aligned with the second to last axis of the right operand, or
with its only axis. The stack axes are aligned as in a binary operation. `__rmatmul__`
is added. This corrects D3.

**Creation.** `from_values_and_labels` accepts each array for which
`array_api_compat.is_array_api_obj` returns `True`. The overloads for NumPy and JAX
stay, and a general overload returns `EagerFrameLabeledArray[..., Any]`. This corrects
D5.

**`from_frame`.** The eager function gets an argument `array_namespace`, with NumPy as
the default. It converts the value column with `to_numpy` and then with
`array_namespace.asarray`.

**Other items.**

1. Add `array-api-strict` to the `dev` dependency group and to the `value_array_type`
   fixture.
2. Add `polder/exceptions.py` and replace the bare exceptions, which corrects D9.
3. Export `align` from the `polder` package, which corrects D10.
4. Change `polder/operations/unary.py` and `polder/operations/align.py` to
   `singledispatch` functions.

### Stage 2: accessors and conversions

1. Add `dtype()`, `ndim()`, `size()` and `device()` as methods to the protocol and to
   both implementations.
2. Add `T` as a property that reverses the axes of an array with two dimensions.
3. Add `astype`, `to_device`, and the scalar conversions `__bool__`, `__int__`,
   `__float__`, `__complex__` and `__index__`.
4. Add `zeros_like`, `ones_like`, `full_like` and `empty_like`.

`ndim()` is a method although it never starts a computation. The reason is that one rule
for all shape accessors is simpler to remember than a rule per accessor. `T` is a
property because `array.T` is the form that each NumPy user knows, and a method call
`array.T()` has no benefit.

The lazy implementation gets the accessors and the scalar conversions in this stage as
well, because they are part of the protocol. `dtype()` reads `collect_schema()` of the
value frame and converts the Narwhals data type to a NumPy data type with a table in
`polder/lazy/dtypes.py`.

### Stage 3: elementwise functions

1. Add the function form of each binary operator, such as `add`, `multiply`, `equal` and
   `bitwise_and`. Each one is the existing function from `eager/binary.py` under the
   array API name.
2. Add the binary functions without an operator: `atan2`, `hypot`, `copysign`,
   `logaddexp`, `maximum`, `minimum`, `nextafter`, `logical_and`, `logical_or` and
   `logical_xor`. `_generate_binop` gets the same form as `_generate_unop`, with a
   function that selects the operation from the array namespace.
3. Add `clip` and `where`. A new helper `align_operands(*operands)` aligns up to three
   arrays and passes scalars through, and `_perform_binop` uses it as well.
4. Add `map_values(array, function)`, which applies a function to the value array and
   checks that the shape of the result is equal.

`map_values` is the planned answer to each array function that polder does not wrap,
such as a function from `jax.scipy.special`. It keeps the number of wrapped functions
limited to the array API, which is the scope that the README states.

### Stage 4: reductions

One generator creates all reductions, in the same way as `_generate_unop`.

```python
def _generate_reduction(op_generator):
    def _perform_reduction(array, /, *, axis=None, keepdims=False):
        axes = normalize_axes(axis, array.ndim())
        values = op_generator(array.array_namespace)(
            array._values, axis=axes, keepdims=keepdims
        )
        labels = tuple(
            None if i in axes else axis_labels
            for i, axis_labels in enumerate(array._labels)
            if keepdims or i not in axes
        )
        return array.from_values_and_labels(values, labels)
    return _perform_reduction
```

1. Add `sum`, `prod`, `mean`, `min`, `max`, `std`, `var`, `all`, `any` and
   `count_nonzero`. The extra arguments of the array API, `dtype` and `correction`, are
   passed to the backend function.
2. Add `argmin` and `argmax`, which accept one axis.
3. Add `cumulative_sum` and `cumulative_prod`, which keep the labels.
4. `axis=None` returns a zero-dimensional array with an empty label tuple.
   `EagerFrameLabeledArray` already accepts this shape.

### Stage 5: axis manipulation

1. Add `permute_dims`, `moveaxis` and `matrix_transpose`. Each one applies the same
   permutation to the value axes and to the label tuple.
2. Add `expand_dims` and `squeeze`. `expand_dims` is equal to an index with `None`.
   `squeeze` raises `ValueError` for a labeled axis, also when its size is 1, because
   the label row would be lost. The user removes such an axis with an integer index.
3. Add `flip`, `roll` and `take`. Each one computes the integer positions per axis one
   time, and applies them to the label frame with `frame[positions, :]` and to the
   values with `index_axis`.
4. Add `take_along_axis`, with the same label rule as `sort`, because each lane selects
   other entries.
5. Add `concat`. It aligns all axes except the concatenation axis, concatenates the
   label frames of that axis with `nw.concat`, and raises `LabelAlignmentError` when
   their columns differ.
6. Add `stack`, with a required `labels` argument that contains one row per input array,
   and `unstack`, which returns a tuple of arrays without that axis.
7. Add `broadcast_arrays`, which is `align` followed by a broadcast of the unlabeled
   axes. A broadcast axis gets the label frame of the array that has labels on it.
8. Add the check for unique labels to the join path of `align` and to `pivot`.

### Stage 6: shape changes

1. Add `diff` with the `label_side` argument. The values come from `xp.diff`. The label
   frame is the slice `[1:]` or `[:-1]` of the input frame.
2. Add `sort` and `argsort` with the `rank_column` argument. For a one-dimensional
   array, the positions from `xp.argsort` reorder the label frame.
3. Add the boolean mask with the full shape, `nonzero_labels` and `unique_values`.
4. Add `searchsorted`, which needs a one-dimensional sorted array and returns integer
   positions with the labels of the searched values.

### Stage 7: linear algebra

1. Add `tensordot`. The `axes` argument selects the axis pairs, and `align` already
   accepts these pairs in its `axes` argument. The result has the remaining label frames
   of the left operand, followed by the remaining label frames of the right operand.
2. Add `vecdot`, which contracts one axis after a full alignment.
3. Leave the `linalg` and `fft` extensions of the array API out. `map_values` covers the
   functions that keep the shape. A function such as `linalg.inv` has a clear label
   rule, so it can be added later on request.

## Lazy implementation

### Lazy test method

Each function has one shared test in `tests/operations` that runs on both
implementations through the existing `implementation` fixture. For the lazy
implementation, the test converts the result with `convert` and compares it to the eager
result with `equals`. The eager implementation is therefore the reference, and the eager
tests against NumPy are the base of that reference.

The tests run with `use_eager_evaluation_for_lazy_arrays` on and off. A new fixture
`frame_backend` runs the lazy tests with Polars, pandas and PyArrow label frames, and
with DuckDB as the lazy engine after stage 8. `_LAZY_UNSUPPORTED` in
`tests/operations/test_unary.py` becomes one list for all functions, so that the missing
functions are visible in one place.

### Stage 8: internal structure

This stage changes the fields of `LazyFrameLabeledArray` before the number of operations
grows. Each later operation then uses the final structure.

**Constructor.** Each operation calls
`type(arr)(labels, values, shape, n_dims, frame_ns)` with all fields. A new field then
needs a change in each operation. The class gets a method `_replace(**changes)` that is
based on `dataclasses.replace`, and all operations use it.

**Index data type.** All index columns, the `axis` column and the `size` column get one
fixed data type, `nw.Int64`. Each place that creates such a column casts it:
`_create_index_df`, each `with_row_index` call, and `upcast_scalar`. This corrects D14.
A signed type is necessary because several plans subtract from an index.

**Deferred checks.** A new field `_checks` holds the conditions that depend on data. See
[Deferred validation](#deferred-validation). `align` adds a check that the label rows
match, which corrects D12. With `auto_align` off, `align` adds a check that the order is
equal and does not reorder, which corrects D11.

**Zero-dimensional arrays.** The value frame has one row and no index column, and the
shape frame has no rows. `_create_index_df` returns a frame with one row and a
placeholder column in that case, because a frame without columns has no rows. This
corrects D13 and is necessary for `axis=None` in the reductions.

**Data types.** After each operation, the `value` column is cast to the data type that
the eager implementation returns. See [Data types](#data-types). A scalar operand is
added with `nw.lit(scalar, dtype)` in the data type of the array, and not as a second
array with a cross join. This corrects D15 and removes one join per scalar operation.

**Label identity.** `align` skips the joins for an axis when the label frames are the
same objects. This is level 1 from [Label identity](#label-identity). Level 2 is not
available, because a comparison of lazy frames needs a collect.

**Lazy engine.** `from_values_and_labels`, `from_frame` and `convert` get an argument
`lazy_backend`. The frames are created with the backend of the label frames and then
converted with `frame.lazy(backend=lazy_backend)`. This corrects D16. `labels()` and
`values()` already collect with `collect(backend=self._frame_ns)`, so the user gets
frames of the original type.

**Values index.** `values[...]` filters the value frame on the selected index positions
before the collect. The dense index frame is then created for the selected shape only.

### Stage 9: protocol parity

**`__getitem__`.** The method handles one axis at a time, like the eager method. Each
index type creates a new label frame for the axis, and a mapping from the old position
to the new position.

| Index         | New label frame                                                     | Values                                                                        |
| ------------- | ------------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| Integer       | Axis removed                                                        | Filter on the position, drop the index column, rename the later index columns |
| Slice         | Filter on the position range, new row index                         | Inner join with the mapping                                                   |
| Integer array | Join with a frame of the positions, which is created from the array | Inner join with the mapping                                                   |
| Mapping       | Filter on the column values. Axis removed when no column remains.   | As an integer or as a slice                                                   |
| Expression    | `filter(expression)`, new row index                                 | Inner join with the mapping                                                   |
| Boolean mask  | Filter on the label rows of the `True` entries of the mask array    | Inner join with the mapping                                                   |
| `None`        | New unlabeled axis                                                  | New index column with the value 0, rename the later index columns             |

See [Index operations](#index-operations) for the details.
`tests/eager/test_indexing.py` moves to `tests/operations` and runs on both
implementations.

**`equals`.** The method returns `False` without a collect when the number of dimensions
or the label columns differ. Otherwise it builds one plan that counts the label rows and
value rows that differ, with the join method of `align`, and collects one number. NaN
values are equal to NaN values, as in the eager method.

**`__matmul__`.** The plan was tested against NumPy.

```python
left.join(right, left_on="__index1", right_on="__index0_right")
    .group_by("__index0", "__index1_right")
    .agg((nw.col("value") * nw.col("value_right")).sum().alias("value"))
```

The contracted axes are aligned first with `align(left, right, axes=((-1, 0),))`, which
already exists.

**Unary functions.** The 17 missing functions are added with the methods from
[Missing expressions](#missing-expressions).

### Stage 10: function parity

The stages 3 to 7 are repeated for the lazy implementation, in the same order. Most
functions follow one of four patterns, and each pattern gets one generator function.

| Pattern                 | Functions                                                            | Lazy plan                                                                                           |
| ----------------------- | -------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| Expression on `value`   | Elementwise functions, `astype`, `clip`, `where`                     | `with_columns` on the value frame, after a join for more than one array                             |
| Change of index columns | `permute_dims`, `moveaxis`, `expand_dims`, `squeeze`, `flip`, `roll` | `rename` or an expression on the `__index` columns and the shape frame                              |
| Aggregation             | Reductions, `matmul`, `tensordot`, `vecdot`                          | `group_by` on the index columns that stay, then an aggregation                                      |
| Window                  | `cumulative_sum`, `cumulative_prod`, `diff`, `sort`, `argsort`       | An expression with `over` on the other index columns and `order_by` on the index column of the axis |

The second pattern has no cost that depends on the number of entries. `permute_dims` is
one `rename` on the value frame, a permutation of the label tuple, and a new `axis`
column in the shape frame. This is an advantage over the eager implementation, where JAX
copies the values.

`concat` adds an offset to the index column of each array after the first. The offset is
the sum of the sizes of the earlier arrays, which comes from their shape frames with a
cross join. The label frames get the same offset, and all frames are combined with
`nw.concat`.

### Stage 11: backends and the third implementation

1. Add DuckDB to the `dev` dependency group and to the `frame_backend` fixture. Correct
   the differences from [Backend coverage](#backend-coverage).
2. Design the third implementation from the README. See
   [Third implementation](#third-implementation).
3. Extend `convert` and the `FrameLabeledArrayImplementation` enumeration.

## Lazy implementation difficulties

### Deferred validation

A lazy operation cannot raise an error for a condition that depends on data, such as
labels that do not match. The shape frame currently has this role: an invalid operation
creates a missing or repeated row, and `shape` then raises "Array has invalid shape".
This message does not name the operation, and some conditions have no effect on the
shape frame, as D12 shows.

The suggested solution is a new field `_checks`, which is a tuple of lazy frames. Each
frame has one row and the columns `message` and `passed`. An operation that can fail
appends one frame. A helper builds the frame from a lazy frame and an aggregation
expression.

```python
def deferred_check(frame, passed: nw.Expr, message: str):
    return frame.select(nw.lit(message).alias("message"), passed.alias("passed"))

# In `align`: each label row of the other array must exist in the reference labels.
check = deferred_check(
    other_labels.join(reference_labels, on=label_columns, how="anti"),
    nw.len() == 0,
    f"Cannot align axis {axis}, because the labels have different rows.",
)
```

`values`, `shape`, `labels`, `equals` and `collect` evaluate
`nw.concat(checks).filter(~nw.col("passed"))` and raise the first message. This plan was
tested with Polars, where it returned the message for labels with different rows.
`collect` stores the result with an empty tuple, so each check runs one time. The result
of an operation contains the checks of all its operands.

The conditions that need a check are:

1. The labels of two arrays contain the same rows. This is the alignment check.
2. The labels of two arrays are in the same order, when `config.auto_align` is `False`.
3. An index with a mapping, such as `{"year": 2021}`, matches exactly one label row when
   it removes the axis.
4. An integer index is within the size of its axis.
5. The label rows are unique, for `align` and `pivot`.
6. The value frame has one row for each index combination. This replaces the null test
   in `values`.

A condition that does not depend on data is still raised when the operation is called.
Examples are a difference in the number of dimensions and a difference in the label
columns, which `collect_schema` returns without a collect.

The rejected alternative is a collect of the label frames in each operation, which gives
an immediate error. The label frames are small compared to the values, but the collect
then runs for each operation in a chain, and a SQL backend sends one query per
operation.

### Index operations

1. An integer, a slice or an integer array with a negative position needs the size of
   the axis. A call to `shape()` collects the shape frame. A cross join with the one-row
   size of the axis keeps the plan lazy. This was tested for the slice `[-2:3]` with
   `nw.when(bound < 0).then(nw.col("size") + bound)`.
2. An index that removes or reorders entries must renumber the `__index` column to the
   range from 0 to the new size. The lazy `pivot` already does this with
   `with_row_index(order_by=...)` on the filtered label frame. The value frame then gets
   the new index with an inner join on the old index. The inner join also removes the
   entries that the index does not select.
3. An integer array index is converted to a frame with the columns `__index` and
   `__new_index` before the plan is built, because the index array is eager data. A
   repeated position creates a repeated label row, which the check for unique labels
   reports in a later alignment.
4. An expression index filters the label frame. This is simpler than in the eager JAX
   case, because the long format has no fixed shape.
5. The number of dimensions after an index is known without a collect, because it only
   depends on the index types and the label columns.
6. A slice with a step is a filter on `(index - start) % step == 0`. A negative step
   also reverses the order in `with_row_index`.
7. The new size of the axis is a count of the new label frame. An unlabeled axis has
   size 1 and only accepts `:`, `0` and `None`, as in the eager implementation.

### Operations that change the shape

`diff` along axis `k` was tested as
`nw.col("value").diff().over(other_index_columns, order_by=index_column)`. The plan then
removes the rows where the index of axis `k` is 0, subtracts 1 from that index in the
value frame and the label frame, and subtracts 1 from the size in the shape frame. With
`label_side="lower"`, the label frame instead loses its last row and keeps its index.

The new shape follows from the old shape with a fixed rule, so no data is needed. The
same holds for `concat`, `stack`, `expand_dims` and `squeeze`. A shape that depends on
data, as for a boolean mask or `unique_values`, is computed as a count of the new label
frame. This is the method of the lazy `pivot`.

The design of the shape frame makes this possible. The shape is a lazy frame and not a
tuple of integers, so an operation can state the new size as an expression on the old
size.

### Order in a lazy frame

A Narwhals `LazyFrame` has no row order. `with_row_index` requires `order_by`, `unique`
has no `maintain_order` argument, and an order-dependent expression such as
`map_batches` raises an error. Each window operation must therefore name its order with
`over(..., order_by=...)`. `cum_sum` and `diff` were tested in this form with Polars and
with DuckDB.

`sort` along an axis needs a rank per lane with a fixed order for equal values.
`rank("ordinal")` has no second sort key, so equal values get an order that is not
defined. The tested solution is
`with_row_index("r", order_by=[*other_index_columns, "value", index_column])`. The new
index is then `r` minus the minimum of `r` over the other index columns. The old index
is the last sort key, so the sort is stable. The old index is also the value of
`argsort`.

### Missing expressions

Narwhals has no expression for `tan`, the inverse trigonometric functions, the
hyperbolic functions, `log1p`, `sign`, `trunc`, `isinf`, `atan2`, `hypot`, `copysign`,
`nextafter`, and `prod` as an aggregation. Bitwise `xor` and the shifts are also
missing. The current code replaces a shift with a multiplication or division by a power
of 2, which has a different overflow behavior. There are three solutions, in order of
preference.

1. Derive the function from existing expressions when the result has the same accuracy.
   Examples are `tan` as `sin / cos`, `sign` and `trunc` with `nw.when`, `isinf` as not
   finite and not NaN, and `hypot` as `sqrt(x**2 + y**2)` with a scale factor against
   overflow.
2. Add a table of native expressions per backend in `polder/lazy/native.py`. The key is
   the function name and the value of `frame.implementation`. The operation converts the
   frame with `to_native`, applies the native expression to the `value` column, and
   wraps the result with `from_native`. This was tested for `arctan` with Polars and for
   `atan` with DuckDB. A backend without an entry raises `NotImplementedError` with the
   name of the backend.
3. Propose the missing expressions to Narwhals, and remove the table entries when a
   release contains them.

The table from solution 2 is necessary for the inverse trigonometric functions, `atan2`,
`log1p`, `expm1` and `prod`. `log1p` and `expm1` have derived forms, `log(1 + x)` and
`exp(x) - 1`, but these lose accuracy for small values. The existing lazy `expm1`
therefore does not have the accuracy of the eager `expm1`. `nextafter` stays
unsupported.

Solution 2 adds backend code to a package that otherwise only uses Narwhals. The
alternative is a function that is missing for all backends, which conflicts with G3. The
table limits the backend code to one module, and each entry is one expression.

A call to a Python function on the values inside the plan is not possible, as
`map_batches` is not allowed on a lazy frame. `map_values` therefore has no lazy
implementation, and the documentation must refer to `convert`.

### Data types

The eager implementation returns the data type of its array backend, and the DataFrame
backends follow other rules. The following results were measured.

| Expression                     | NumPy  | Polars   | DuckDB     |
| ------------------------------ | ------ | -------- | ---------- |
| `max` of `[1.0, nan]`          | `nan`  | `1.0`    | `nan`      |
| `-7 // 2`                      | `-4`   | `-4`     | `-3`       |
| `-7 % 2`                       | `1`    | `1`      | `-1`       |
| `int8` `100 + 100`             | `-56`  | `-56`    | Error      |
| `int8` array plus `1`          | `int8` | `int64`  | Not tested |
| `bool` array plus `bool` array | `bool` | `uint32` | Not tested |

The suggested solution has four parts.

1. A function `result_dtype(operation, *dtypes)` in `polder/lazy/dtypes.py` returns the
   data type of the result. It converts the Narwhals data types to NumPy data types and
   calls `np.result_type`, with exceptions for true division and comparisons. The
   `value` column is cast to this type after each operation.
2. Floor division and remainder are written with expressions that have one definition on
   each backend: `floor(a / b)` for floating point values, and a correction with
   `nw.when` on the sign of the remainder for integers.
3. NaN propagation is made explicit in `min`, `max`, `argmin` and `argmax`, with an
   `is_nan().any()` aggregation and `nw.when`. This plan was tested and returned `nan`
   for a lane with a NaN.
4. Null and NaN stay separate. A null in the `value` column always means that an entry
   is missing. An operation that creates nulls, such as `diff`, must remove those rows
   in the same step.

Integer overflow stays a documented difference. An array backend wraps the value, DuckDB
raises an error, and a plan that copies the wrap behavior costs more than the operation.

### Reductions

A reduction is a `group_by` on the index columns that stay, followed by a rename of the
later index columns. With `keepdims=True`, the reduced index column is added again with
the value 0. Three cases need extra work.

1. A reduction over an axis of size 0 returns no rows, while the eager implementation
   returns the identity value, such as 0 for `sum`. The plan needs a left join from the
   full index of the result and a `fill_null` with the identity value. This join is only
   added for `sum`, `prod`, `all`, `any` and `count_nonzero`, because the other
   reductions have no identity value.
2. `argmin` and `argmax` have no aggregation in Narwhals. The tested plan filters the
   rows where the value equals the extreme value over the lane, and then takes the
   minimum index. This returns the first position for equal values, as NumPy does.
3. `prod` has no aggregation in Narwhals. It needs the native expression table, because
   `exp(sum(log(x)))` loses accuracy and does not handle zero or negative values.

A reduction with `axis=None` is a `select` with the aggregation, without `group_by`.

### Plan size

Each binary operation adds one join per labeled axis for alignment, and one join for the
values. In the measurement, a chain of 8 additions on a 300 by 300 array took 603 ms. A
hash join on all index columns costs more than the same elementwise operation on an
array, and this conflicts with G4 for dense numeric code. Four measures limit the cost.

1. Skip alignment for label frames that are the same objects, as in stage 8.
2. Add a scalar operand as a literal, as in stage 8.
3. Document `collect` as the method that ends the current plan and stores its result.
4. State in the user guide that the lazy implementation fits relational work, and that
   `convert` to the eager implementation fits dense numeric loops.

The value join stays necessary for two arrays with the same label objects, because the
rows of two lazy frames have no common order.

### Methods that return Python values

`equals`, `shape`, `__bool__`, `__int__`, `__float__` and `__index__` return Python
values, so each call collects the plan. This matches the statement in the README that a
computation is resolved when its values are converted to an eager form. The
documentation of each method must state that the call collects.

`shape` is the most frequent of these calls. In the existing operations, the shape frame
is built from label frames and other shape frames only, so its collect does not read the
value frame. New operations must keep this property.

### Backend coverage

The lazy implementation was run with three label frame types.

| Label frames | Result                                                                                     |
| ------------ | ------------------------------------------------------------------------------------------ |
| Polars       | All existing tests pass.                                                                   |
| pandas       | Creation, binary operation with alignment, `pivot` and `unpivot` returned correct results. |
| PyArrow      | Creation works. All other operations fail with D14.                                        |

DuckDB was tested as a lazy engine on the Narwhals level, with frames from
`DataFrame.lazy(backend="duckdb")`. Window expressions with `order_by`, `group_by`,
`concat` and `with_row_index` returned correct results. Two plans raised a DuckDB
`BinderException` about a column reference: a cross join, and an anti-join followed by a
`select` of a literal. Both forms are in use in the lazy implementation, so stage 11
must find the cause and a form that works on both backends.

## Third implementation

G6 describes an implementation that holds an eager value array and applies index
operations lazily. The proposed structure reuses parts of both existing implementations.

| Field             | Content                                                                                                    |
| ----------------- | ---------------------------------------------------------------------------------------------------------- |
| `_values`         | The value array, as in the eager implementation                                                            |
| `_indexed_labels` | One lazy label frame per axis, with an `__index` column, as in the lazy implementation                     |
| `_positions`      | One lazy frame per axis, or `None`. The frame maps each position of the result to a position in `_values`. |

An index, `flip`, `roll`, `take` and an alignment only change `_indexed_labels` and
`_positions`. An operation that needs the values first applies each pending frame in
`_positions` with one `xp.take` per axis, and then calls the eager operation. A chain of
an index, an alignment and a second index therefore copies the values one time.

This structure matches the flow from the README, in which a transformation step, a
computation step and a second transformation step follow each other. `pivot` and
`unpivot` need more than a mapping per axis, so they apply the pending positions first.
The design is a separate task after stage 10, because it can then reuse the index plans
from stage 9.

## Stage order

The order of the stages follows from three rules.

1. A defect is corrected before a function is added on top of it. Stage 1 and stage 8
   therefore contain no new functions.
2. Each eager stage is complete before the lazy stage for the same functions, because
   the eager result is the reference in the lazy tests.
3. Inside each implementation, the functions with a simple label rule come first. The
   functions that change the shape and need new arguments come later, so that the open
   questions about their labels do not block the other work.

Stage 8 does not depend on stages 2 to 7. It can start when stage 1 is complete, if the
lazy defects have a higher priority than the new eager functions.

## Open questions

1. Should the functions with the name of a Python builtin get a trailing underscore, as
   `abs_` and `round_` have? The alternative is `pld.sum` and `pld.abs` without an
   underscore, with the old names as aliases.
2. Should a function return the label row of the extreme entry, in addition to `argmin`
   and `argmax`? A position is only valid for the label order of one array.
3. Should the reductions accept label columns in addition to an axis number, such as
   `sum(array, over={0: ["year"]})`? This is equal to a `pivot` followed by a reduction,
   and it avoids the dense intermediate array.
4. Should the check for unique labels run at creation, with a configuration setting to
   disable it, instead of in `align` and `pivot`?
5. Is `rank` the correct default label for a sorted axis, or should `sort` on an array
   with more than one dimension be an error?
6. Is a table of native expressions per backend acceptable in the lazy implementation?
   Without it, about 12 functions stay unsupported until Narwhals adds the expressions.
7. Should integer overflow in the lazy implementation stay a documented difference
   between backends?

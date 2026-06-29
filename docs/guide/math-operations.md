# Mathematical operations

A frame-labeled array is an array first, so it supports the mathematical operations of the
[array API](https://data-apis.org/array-api/latest/). The values are transformed by the
backend, while the labels are carried along. For binary operations the operands are also
[aligned](alignment.md), so that you only ever combine values that carry the same labels.

## Elementwise binary operators

All the array API binary operators are available as the usual Python operators. They work
between two arrays and between an array and a scalar.

```python
import numpy as np
import polars as pl

import polder as pld

labels = [pl.DataFrame({"i": [0, 1]}), pl.DataFrame({"j": [0, 1]})]
a = pld.from_values_and_labels(np.array([[1.0, 2.0], [3.0, 4.0]]), labels)
b = pld.from_values_and_labels(np.array([[5.0, 6.0], [7.0, 8.0]]), labels)

c = a + b        # elementwise, after aligning a and b
d = a * 2.0      # scalar broadcasts across the array
e = 2.0 - a      # reflected operators work too

np.testing.assert_array_equal(c.values(), a.values() + b.values())
```

The full set is the arithmetic operators (`+`, `-`, `*`, `/`, `//`, `%`, `**`), the bitwise
operators (`&`, `|`, `^`, `<<`, `>>`), and the comparison operators (`<`, `<=`, `>`, `>=`,
`==`, `!=`). Comparisons return an array of booleans, with the labels preserved.

```{.python continuation}
mask = a < b     # a boolean-valued frame-labeled array
assert mask.values().all()
```

When one operand is an array and the other is a scalar, the scalar broadcasts across the
whole array and the array's labels are preserved unchanged. When both operands are arrays,
they are aligned first (see [Alignment](alignment.md)), so their labels must be compatible.

## Unary operators

The unary operators `+`, `-`, `abs()`, and `~` are supported directly.

```{.python continuation}
neg = -a
mag = abs(a)
assert mag.values().min() >= 0
```

## Elementwise functions

Beyond the operators, the array API's unary elementwise functions are available as
functions in the top-level package. They take an array and return an array with the same
labels and transformed values.

```{.python continuation}
s = pld.sin(a)
r = pld.sqrt(a)
l = pld.log(a)

np.testing.assert_allclose(s.values(), np.sin(a.values()))
```

The available functions cover the array API unary set, including the trigonometric
functions (`sin`, `cos`, `tan` and their inverses and hyperbolic variants), exponentials
and logarithms (`exp`, `expm1`, `log`, `log1p`, `log2`, `log10`), rounding (`ceil`, `floor`,
`round_`, `trunc`), sign and magnitude (`abs_`, `sign`, `signbit`, `reciprocal`, `square`,
`sqrt`), the classification predicates (`isfinite`, `isinf`, `isnan`), and the bitwise and
logical inversions (`invert`, `bitwise_invert`, `logical_not`). The complete list with
signatures is in the [API reference](../api/operations.md).

A few of these functions are named with a trailing underscore (`abs_`, `round_`) to avoid
clashing with Python builtins, following the array API.

!!! note "Backend coverage"

    Every unary function is supported on [eager arrays](eager-arrays.md). A subset is also
    implemented for [lazy arrays](lazy-arrays.md). If you call one that the lazy
    implementation does not yet provide, it raises `NotImplementedError`. You can
    [convert](conversion.md) to the eager implementation to get the full set.

## Matrix multiplication

The matrix multiplication operator `@` performs a matrix product on the values, and merges
the labels appropriately: it takes all but the last axis from the left operand and all but
the first axis from the right operand. The shared inner axis is contracted away, exactly as
its labels would suggest.

```{.python continuation}
rng = np.random.default_rng(0)

left = pld.from_values_and_labels(
    rng.standard_normal((2, 3)),
    [pl.DataFrame({"i": np.arange(2)}), pl.DataFrame({"j": np.arange(3)})],
)
right = pld.from_values_and_labels(
    rng.standard_normal((3, 4)),
    [pl.DataFrame({"j": np.arange(3)}), pl.DataFrame({"k": np.arange(4)})],
)

product = left @ right
assert product.shape() == (2, 4)   # labeled by "i" and "k"
```

As with the elementwise operators, the operands are aligned over the contracted axis first,
so shuffling the labels of the inner axis makes no difference to the result.

!!! note "Eager only for now"

    Matrix multiplication is currently supported on eager arrays. On lazy arrays `@` raises
    `NotImplementedError`.

## Equality of arrays

The `==` operator is elementwise and returns a boolean array, following NumPy. To ask
whether two arrays are equal as a whole, both their values and their labels, use the
`equals` method instead. Because `equals` compares labels, two arrays that hold the same
data in a different label order are only equal after [alignment](alignment.md).

```{.python continuation}
from polder.operations.align import align

assert not a.equals(b)   # whole-array equality, values and labels

# The same data as a, but with the first axis reversed.
shuffled_a = pld.from_values_and_labels(
    a.values()[::-1],
    [pl.DataFrame({"i": [1, 0]}), pl.DataFrame({"j": [0, 1]})],
)

aligned_a, aligned_shuffled = align(a, shuffled_a)
assert aligned_a.equals(aligned_shuffled)
```

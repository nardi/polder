# Operations

Generic operations defined for any frame-labeled array. The unary functions are all
re-exported at the top level, so they are available as `pld.sin`, `pld.sqrt`, and so on.

## Unary elementwise functions

These follow the [array API](https://data-apis.org/array-api/latest/) unary set. Each takes
a single frame-labeled array and returns an array with the same labels and transformed
values, with the signature `f(arr) -> arr`. See
[Mathematical operations](../guide/math-operations.md) for an introduction. A few names end
in a trailing underscore (`abs_`, `round_`) to avoid clashing with Python builtins.

Every function below is supported on [eager arrays](../guide/eager-arrays.md). The ones
marked in the *Lazy* column are also implemented for
[lazy arrays](../guide/lazy-arrays.md). The others raise `NotImplementedError` on a lazy
array until they are implemented.

The *Lazy* column says "yes" where a lazy implementation exists today.

### Sign and magnitude

| Function | Meaning | Lazy |
| --- | --- | --- |
| `pos(arr)` | Unary plus (`+arr`). | yes |
| `neg(arr)` | Negation (`-arr`). | yes |
| `abs_(arr)` | Absolute value. | yes |
| `sign(arr)` | Sign of each element. | no |
| `signbit(arr)` | Whether the sign bit is set. | yes |
| `reciprocal(arr)` | Elementwise `1 / arr`. | yes |
| `square(arr)` | Elementwise `arr ** 2`. | yes |
| `sqrt(arr)` | Square root. | yes |

### Rounding

| Function | Meaning | Lazy |
| --- | --- | --- |
| `ceil(arr)` | Round up to the nearest integer. | yes |
| `floor(arr)` | Round down to the nearest integer. | yes |
| `round_(arr)` | Round to the nearest integer. | yes |
| `trunc(arr)` | Round toward zero. | no |

### Exponentials and logarithms

| Function | Meaning | Lazy |
| --- | --- | --- |
| `exp(arr)` | Exponential. | yes |
| `expm1(arr)` | `exp(arr) - 1`, accurate for small values. | yes |
| `log(arr)` | Natural logarithm. | yes |
| `log1p(arr)` | `log(1 + arr)`, accurate for small values. | no |
| `log2(arr)` | Base-2 logarithm. | yes |
| `log10(arr)` | Base-10 logarithm. | yes |

### Trigonometric and hyperbolic

| Function | Meaning | Lazy |
| --- | --- | --- |
| `sin(arr)`, `cos(arr)`, `tan(arr)` | Trigonometric functions. | `sin`, `cos` |
| `asin(arr)`, `acos(arr)`, `atan(arr)` | Inverse trigonometric functions. | no |
| `sinh(arr)`, `cosh(arr)`, `tanh(arr)` | Hyperbolic functions. | no |
| `asinh(arr)`, `acosh(arr)`, `atanh(arr)` | Inverse hyperbolic functions. | no |

### Classification and complex numbers

| Function | Meaning | Lazy |
| --- | --- | --- |
| `isfinite(arr)` | Whether each element is finite. | yes |
| `isinf(arr)` | Whether each element is infinite. | no |
| `isnan(arr)` | Whether each element is NaN. | yes |
| `real(arr)` | Real part of a complex array. | no |
| `imag(arr)` | Imaginary part of a complex array. | no |
| `conj(arr)` | Complex conjugate. | no |

### Bitwise and logical

| Function | Meaning | Lazy |
| --- | --- | --- |
| `invert(arr)` | Bitwise inversion (`~arr`). | yes |
| `bitwise_invert(arr)` | Bitwise inversion, named form. | yes |
| `logical_not(arr)` | Logical NOT. | yes |

## Alignment

The `align` function reorders arrays so that their labels match. See
[Alignment](../guide/alignment.md) for an introduction.

::: polder.operations.align.align

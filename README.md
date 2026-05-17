<div align="center">

  <h2 align="center">
  <code>polder</code> - Finding the middle-ground between array and relational data models
  </h2>

</div>

## Introduction

`polder` is a Python package that provides a simple composite data container type:

1. a Numpy-style ([array API](https://data-apis.org/array-api/latest/)) array,
2. whose entries are labeled using Polars-compatible ([Narwhals](https://narwhals-dev.github.io/narwhals/)) DataFrames.

In other words, it lets you bundle your *data* and your *metadata* into a single container, using an array-based library to handle the data, and a relational library to handle the metadata.

### Why?

The common situation in which this is useful if you have a Numpy(-like) array, but find yourself having difficulties keeping track of what the values represent. You might find yourself creating an auxillary data structure for this, where you keep some list of labels next to your array, so that you avoid for example combining two arrays with the same values, but in a different order. Then, you might end up having to write extra code keeping track of these labels throughout your computation. `polder` aims to provide this data structure in a more general way, that you can adapt to the needs of your application.

This is similar to the [pandas](https://pandas.pydata.org/) idea of a DataFrame, as a Numpy array with labels. With pandas you can take a 2D Numpy array and with it's `Index`es apply arbitrarily complex labels on both axes (rows and columns). However it falls short in a number of ways:

1. Its API is quite large and, while intuitive for simple tasks, provides a lot of surprises as well.
2. While its `MultiIndex` technically supports very complex labeling, this is not the most intuitive part of the API and its functionality is often not effectively used.
3. It only supports 2-dimensional data.
4. It is not very performance-minded: many of its operations (especially around the index) perform "magic" such as automatically reordering data or extending it with missing values. This also makes it hard to predictably interact with other libraries without a full conversion step.

We are of the opinion the relational model of the DataFrame as promoted by Polars and similar libraries is the better fit, and as such this library aims to maintain a clear separation between the array part and the relational part.

It is also quite similar to [Xarray](https://docs.xarray.dev/en/stable/), which provides a very flexible "labeled array" container, but there are some conceptual differences. These lie mostly in the intended scope of the library, which leads to some different design decisions:

1. `polder` aims to be purely an orchestration layer over existing array and data processing libraries, so that all significant processing work is dispatched to those, in order to make use of their efficient implementations.
2. `polder` aims to provide a simple data model, which allows converting array-based code without having to reconsider your data modelling approach. In other words, you should be able to take an unlabeled array, and it should be "obvious" from context which labels to assign to the entries.
3. `polder` aims to support a larger range of array/DataFrame backends, in order to enable the use of the library in various contexts (for example, as part of adding structure to a GPU-backed numerical algorithm, or to perform automatic differentiation through a labeled computation).
4. `polder` aims to not have any performance surprises compared to the backend libraries. For example, if you have a Numpy-backed array, doing `array[3:5]` should return a view, doing `array[[5, 7]]` should return a copy. The only exception to this is that many operations perform *alignment*: `arr1 + arr2` will ensure that the labels for both are aligned, otherwise the result is not meaningful. This auto-alignment can be disabled, in case you want to avoid all unexpected performance regressions.
5. `polder` doesn't try to provide a universal data model: if you have data that doesn't nicely fit into an array structure or you want to perform certain operations that are not well defined in terms of array shapes (such as binning), you might be better off just using a DataFrame library.

## Status

At the center of the library is the `FrameLabeledArray` protocol, which is what user code should be written against. This protocol currently supports a number of basic functionalities:
- Decomposing into `values` (array-like) and `labels` (sequence of DataFrame-likes).
- Indexing (NumPy-style), which also supports indexing using Narwhals expressions.
- Pivoting: splitting a single axis into multiple, by orthogonally decomposing the labels into seperate dimensions corresponding to different label columns (and the reverse, "unpivoting").
- All special (dunder) operations as defined in the [array API](https://data-apis.org/array-api/latest/).

This protocol is supported by a number of generic operations:
- Creation: `pld.from_values_and_labels`, `pld.from_frame`
- Alignment: reordering multiple arrays so that their labels match up (this usually happens automatically when performing other operations).
- Unary elementwise operations as defined in the [array API](https://data-apis.org/array-api/latest/) (e.g. `pld.sin(arr)`).

Then, there are currently two implementations of the protocol:

1. An implementation that is eager (every operation is fully resolved immediately) and in principle supports any array that follows the [array API](https://data-apis.org/array-api/latest/). Currently there is "real" support (i.e. with tests and proper typing) for NumPy and JAX arrays. Note that JAX is an optional dependency, so if you are using the library you have to install it manually, but this is usually already the case (how else would you pass in a JAX array?). This implementation supports all functionality defined in the protocol.
2. An implementation that is entirely backed by Narwhals LazyFrames, and evaluates all operations lazily. This is useful when you only want to express your operations in an array-style and don't need any special interop with array-based libraries, but either don't care about the way the values are stored, or already have your data in a DataFrame format supported by Narwhals and would like to keep all operations within that format. Using this implementation you can keep your data in the same backend it already is, and use whatever query engine that backend has to optimize large computations. This implementation is in early development and does not yet support all functionality.

The advantage of having multiple implementations is also that you can convert easily between them. It might make sense to first perform some processing fully in the relational context, so more of the computation can be efficiently handled by a single query engine, and then switch over to the hybrid model when some interop with array-libraries is important.

Ideas for future implementations are:

1. An implementation that acts eagerly on arrays, but performs lazy indexing/reordering of values. This will make use of Narwhals lazy API for the labels, allowing to e.g. perform multiple slicing/reshaping operations without making excess copies of the array values. Since a lot of code has a "transformation → computation → transformation" flow, where the transformation steps are mostly reordering data but not changing the values, and the computation wants to keep the data in its most efficient shape, this handoff can be a natural transition point between the relational operations on the metadata and the array-based operations in the computational core.

## Development guidelines

There are a few structural guidelines for the development of this library.

### The labeled array protocol

The main data container is `FrameLabeledArray`, which is provided as a protocol. The idea is that users can write code against this protocol, and then this code can be run against various implementations of the protocol.

To be more flexible in code reuse vs specialization for efficiency, the various implementations don't inherit from a common ancestor. Instead they all implement the protocol, but are free to vary in implementation arbitrarily. As such the decision whether something should be in the protocol or not is quite significant, as it should generalize to all implementations, and it will also mean that users will write code against it. For those reasons stability of the protocol is quite important.

The protocol should mostly follow the [array API](https://data-apis.org/array-api/latest/), since the labeled array objects are arrays first. Additional functionality (such as `align` or `pivot`) are defined if they make sense for the format. Some array API functionality is also extended, such as indexing with an expression (that will filter the labels) or reshaping along label dimensions instead of giving an explicit shape.

To support all kinds of backend libraries, the protocol is limited to only those functionalities that make sense in every paradigm. Particularly, all arrays are immutable, with the expectation that for high performance code an accelerator library is used. In addition, because the degree of "laziness" may vary between implementations, this cannot easily be made explicit in the protocol. Instead you should expect computations to be resolved as soon as the array values are converted to an eager form, for example when you call `array.values()`.

Because of the heavy reliance on protocols/structural typing, all library code should be fully typed.

### Code structure

The structure of the repo is roughly as follows:

* The generic protocols are defined in the `protocols` subpackage.
* Implementations get their own subpackage, for example anything related to the eager implementation is stored in the `eager` subpackage.
* Generic operations defined for `FrameLabeledArray` (that either have a generic implementation or dispatch to more specialized ones based on type) are defined in the `operations` subpackage.
* Common operations are re-exported in the top-level `polder` package for ease of use.

### Style

* All code is autoformatted using Ruff.
* Tests are written using pytest.
* Specific backend libraries should not be requirements for end-users, they should always be dynamically imported, with a few exceptions (currently `narwhals` and `numpy`).
* The recommended usage is `import polder as pld` (similar to other libraries).
* When writing comments, please use full sentences and always end with a period. Comments that are a single phrase don't need a period (think section header). Always put comments on a separate line, not after code.

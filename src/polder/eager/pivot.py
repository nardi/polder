from __future__ import annotations

import itertools
import math
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, TypeAlias, cast

import array_api_extra as xpx
import narwhals as nw
import narwhals.typing as nwt
import numpy as np
from typing_extensions import TypeAliasType

from polder.eager.value_array import SomeValueArray, ValueArrayNamespace
from polder.protocols.array import AnyDataFrame

if TYPE_CHECKING:
    from polder.eager.array import SomeEagerFrameLabeledArray

_NO_FILL = object()

AxisLabelsSpecifier: TypeAlias = Sequence[str]
AxisLabelsToPivot: TypeAlias = AxisLabelsSpecifier | Sequence[AxisLabelsSpecifier]
AxesSlice = tuple[int, int]
Labels = TypeAliasType(
    "Labels", Sequence[nwt.DataFrameT | None], type_params=(nwt.DataFrameT,)
)


def pivot(
    arr: SomeEagerFrameLabeledArray,
    /,
    *,
    axis_labels_to_pivot: Mapping[int, AxisLabelsToPivot],
    fill_value: Any = _NO_FILL,
) -> SomeEagerFrameLabeledArray:
    """Pivots one or more label columns into a new axis.

    The `pivot` function can be used to split an axis into multiple orthogonal axes. Suppose an axis
    `i` is labeled by three columns, `x`, `y`, and `t`. If we know that we have a value for each
    unique combination of `(x, y)` and `t`, we might want to make this structure explicit in the
    array by splitting `i` into two axes `j` and `k`, where `j` is labeled by `x` and `y`, and `k`
    is labeled by `t`. To do so, we can use `pivot`:

    ```python
    arr2 = arr.pivot(axis_labels_to_pivot={i: ["t"]})
    ```

    This will create a new axis in `arr2` at location `i + 1`, which is labeled by `t`. If you want
    to separate an axis into more than two axes, this is also supported:

    ```python
    arr2 = arr.pivot(axis_labels_to_pivot={i: [["y"], ["t"]]})
    ```

    You can also split multiple label columns into a new axis together. Any label columns not
    mentioned, will be kept in the first axis.

    For a pivot operation to work, the array needs to contain a value for each combination in the
    product of the split labels (in this case, `(x, y)` and `t`). If this is not the case, you can
    use `fill_value` to fill unknown values. Because the labels in the input array are unique, no
    information will be lost in any case.
    """
    values = arr.values()
    labels = tuple(arr.labels())
    xp = arr.array_namespace

    # Pivot each axis one at a time, processing in descending order so earlier pivots do not affect
    # axis indices of later pivots.
    for axis in sorted(axis_labels_to_pivot.keys(), reverse=True):
        pivot_spec = axis_labels_to_pivot[axis]
        values, labels = _pivot_single_axis(
            xp, values, labels, axis, pivot_spec, fill_value
        )

    return type(arr)(tuple(labels), values)


def _pivot_single_axis(
    xp: ValueArrayNamespace[SomeValueArray],
    values: SomeValueArray,
    labels: Labels,
    axis: int,
    pivot_spec: AxisLabelsToPivot,
    fill_value: Any,
) -> tuple[SomeValueArray, Labels]:
    """Pivot a single axis, potentially creating one or more new axes."""
    # Normalize pivot_spec to a list of pivot groups, each specifying columns to pivot together.
    pivot_groups = _normalize_pivot_spec(pivot_spec)

    # Handle each pivot group in reverse order (so that unspecified columns are implicitly first in
    # the axis order).
    for pivot_cols in reversed(pivot_groups):
        values, labels = _pivot_single_group(
            xp, values, labels, axis, pivot_cols, fill_value
        )

    return values, labels


def _normalize_pivot_spec(spec: AxisLabelsToPivot) -> list[list[str]]:
    """Normalize pivot specification to a list of lists of column names."""
    # If spec is a sequence of strings, it's a single group.
    if all(isinstance(s, str) for s in spec):
        spec = cast(list[str], list(spec))
        return [spec]
    # Otherwise, it's a sequence of groups.
    return [list(group) for group in spec]


def _pivot_single_group(
    xp: ValueArrayNamespace[SomeValueArray],
    values: SomeValueArray,
    labels: Labels,
    axis: int,
    pivot_cols: list[str],
    fill_value: Any,
) -> tuple[SomeValueArray, Labels]:
    """Pivot a single group of columns into a new axis."""

    labels_df = labels[axis]
    if labels_df is None:
        raise ValueError(f"Cannot pivot unlabeled axis {axis}")

    # Determine which columns to keep vs which to pivot.
    keep_cols = [col for col in labels_df.columns if col not in pivot_cols]

    # If there are no columns to keep, that means we are already done.
    if not keep_cols:
        return values, labels

    # Add a row index to track original positions.
    labels_with_idx = labels_df.with_row_index("__idx")

    # Determine unique combinations of keep columns (these become the new axis labels).
    keep_labels = labels_df.select(keep_cols).unique(maintain_order=True)

    # Determine unique combinations of pivot columns (these become the new axis to insert).
    pivot_labels = labels_df.select(pivot_cols).unique(maintain_order=True)

    # Build a cross join to get all (keep, pivot) combinations, then join with original data to find
    # the row indices.
    all_combinations = (
        keep_labels.with_row_index("__keep_idx")
        .lazy()
        .join(pivot_labels.with_row_index("__pivot_idx").lazy(), how="cross")
    )
    all_combinations = all_combinations.join(
        labels_with_idx.lazy(), on=keep_cols + pivot_cols, how="left"
    )

    # Sort to ensure consistent ordering by keep and pivot indices. Since we are sorting on the
    # numerical index columns we added, this is always possible.
    all_combinations = all_combinations.sort("__keep_idx", "__pivot_idx")

    # Extract the row indices.
    row_idx = all_combinations.select("__idx").collect()

    # Check for missing values before processing.
    has_missing = row_idx.select(nw.all().is_null().any()).item()
    if has_missing and fill_value is _NO_FILL:
        raise ValueError("Missing pivot combinations and no `fill_value` provided.")

    # Convert the indices to an 2D array for indexing into the values array.
    value_idx = row_idx["__idx"].to_numpy().reshape(len(keep_labels), len(pivot_labels))

    if has_missing:
        missing_mask = np.isnan(value_idx.astype(float))
        missing_idx = xp.asarray(np.argwhere(missing_mask))
        # Replace missing indices with 0 temporarily for indexing (these will later be filled with
        # `fill_value`).
        value_idx_valid = np.where(missing_mask, 0, value_idx).astype(int)
        new_values = xp.take(values, xp.asarray(value_idx_valid), axis=axis)
        # Fill missing indices with `fill_value`.
        # NOTE: the protocols from types-array-api and array-api-extra are not
        # compatible. We just ignore the typing here for now.
        new_values = xpx.at(new_values)[  # type: ignore
            (slice(None),) * axis + (missing_idx[:, 0], missing_idx[:, 1])
        ].set(fill_value)
        new_values = cast(SomeValueArray, new_values)
    else:
        # All combinations present, just index the values array.
        value_idx_valid = value_idx.astype(int)
        new_values = xp.take(values, xp.asarray(value_idx_valid), axis=axis)

    # Insert labels in the correct place.
    new_labels = [*labels[:axis], keep_labels, pivot_labels, *labels[axis + 1 :]]

    return new_values, new_labels


def unpivot(
    arr: SomeEagerFrameLabeledArray,
    /,
    *,
    axes_to_merge: Sequence[AxesSlice],
) -> SomeEagerFrameLabeledArray:
    """Merge multiple consecutive axes into one.

    The reverse of the `pivot` operation, as that splits a single axis into
    multiple by separating them along the label columns. `unpivot` can recombine
    these axes into a single one, taking the product of all labels.

    Axes are specified as inclusive "psuedo-slices" in `axes_to_merge`. For
    example the following will merge axes 1, 2, and 3 into a single axis, as
    well as 5 and 6:

    ```python
    arr2 = arr.unpivot(axes_to_merge=[(1, 3), (5, 6)])
    ```

    The slices must have no overlap, i.e. `(1, 3)` and `(2, 4)` are not allowed
    together. Also, to avoid mistakes they must be provided in order.
    """
    # Validate `axes_to_merge`.
    flat_axes_to_merge = itertools.chain(*axes_to_merge)
    if not all(a1 < a2 for a1, a2 in itertools.pairwise(flat_axes_to_merge)):
        raise ValueError(
            f"Axis slices are invalid, either unordered, single-valued, or not disjoint: {axes_to_merge}"
        )

    # Extract shape and labels.
    shape = arr.shape()
    labels = arr.labels()

    for first_axis, last_axis in reversed(axes_to_merge):
        axes_labels = labels[first_axis : last_axis + 1]
        unlabeled_count = sum(1 for axis_labels in axes_labels if axis_labels is None)
        if unlabeled_count:
            if unlabeled_count != len(axes_labels):
                raise ValueError("Cannot merge unlabeled axes with labeled ones.")

            # All axes are unlabeled, so we can simply merge them together into
            # one unlabeled axis.
            merged_axis_labels = None
        else:
            axes_labels = cast(list[AnyDataFrame], axes_labels)

            # Create the merged labels as a cross product of the axes. Add row
            # numbers to ensure the result has the correct (row-major) order.
            row_index_columns = tuple(f"__index{i}" for i in range(len(axes_labels)))
            merged_axis_labels = (
                axes_labels[0].with_row_index(row_index_columns[0]).lazy()
            )
            for i, axis_labels in enumerate(axes_labels[1:]):
                merged_axis_labels = merged_axis_labels.join(
                    axis_labels.with_row_index(row_index_columns[i + 1]).lazy(),
                    how="cross",
                )
            merged_axis_labels = (
                merged_axis_labels.sort(row_index_columns)
                .drop(row_index_columns)
                .collect()
            )

        # Replace the existing labels by the merged ones.
        labels = (*labels[:first_axis], merged_axis_labels, *labels[last_axis + 1 :])
        # Replace the corresponding shape entries by their product.
        shape = (
            *shape[:first_axis],
            math.prod(shape[first_axis : last_axis + 1]),
            *shape[last_axis + 1 :],
        )

    # Extract and reshape the values.
    values = arr.array_namespace.reshape(arr.values(), shape)

    return type(arr)(labels, values)

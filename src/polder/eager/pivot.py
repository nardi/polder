from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, TypeAlias, cast

import narwhals as nw
import narwhals.typing as nwt
import numpy as np

if TYPE_CHECKING:
    from polder.eager.array import SomeEagerFrameLabeledArray

_NO_FILL = object()

AxisLabelsSpecifier: TypeAlias = Sequence[str]
AxisLabelsToPivot: TypeAlias = AxisLabelsSpecifier | Sequence[AxisLabelsSpecifier]
Labels = Sequence[nwt.DataFrameT | None]


def pivot(
    arr: SomeEagerFrameLabeledArray,
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
    arr2 = pivot(arr, axis_labels_to_pivot={i: ["t"]})
    ```

    This will create a new axis in `arr2` at location `i + 1`, which is labeled by `t`. If you want
    to separate an axis into more than two axes, this is also supported:

    ```python
    arr2 = pivot(arr, axis_labels_to_pivot={i: [["y"], ["t"]]})
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

    # Pivot each axis one at a time, processing in descending order so earlier pivots do not affect
    # axis indices of later pivots.
    for axis in sorted(axis_labels_to_pivot.keys(), reverse=True):
        pivot_spec = axis_labels_to_pivot[axis]
        values, labels = _pivot_single_axis(
            values, labels, axis, pivot_spec, fill_value
        )

    return type(arr)(tuple(labels), values)


def _pivot_single_axis(
    values: np.ndarray,
    labels: Labels,
    axis: int,
    pivot_spec: AxisLabelsToPivot,
    fill_value: Any,
) -> tuple[np.ndarray, Labels]:
    """Pivot a single axis, potentially creating one or more new axes."""
    # Normalize pivot_spec to a list of pivot groups, each specifying columns to pivot together.
    pivot_groups = _normalize_pivot_spec(pivot_spec)

    # Handle each pivot group in reverse order (so that unspecified columns are implicitly first in
    # the axis order).
    for pivot_cols in reversed(pivot_groups):
        values, labels = _pivot_single_group(
            values, labels, axis, pivot_cols, fill_value
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
    values: np.ndarray,
    labels: Labels,
    axis: int,
    pivot_cols: list[str],
    fill_value: Any,
) -> tuple[np.ndarray, Labels]:
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
        missing_idx = np.argwhere(missing_mask)
        # Replace missing indices with 0 temporarily for indexing (these will later be filled with
        # `fill_value`).
        value_idx_valid = np.where(missing_mask, 0, value_idx).astype(int)
        new_values = np.take(values, value_idx_valid, axis=axis)
        # Fill missing indices with `fill_value`.
        new_values[(slice(None),) * axis + (missing_idx[:, 0], missing_idx[:, 1])] = (
            fill_value
        )
    else:
        # All combinations present, just index the values array.
        value_idx_valid = value_idx.astype(int)
        new_values = np.take(values, value_idx_valid, axis=axis)

    # Insert labels in the correct place.
    new_labels = [*labels[:axis], keep_labels, pivot_labels, *labels[axis + 1 :]]

    return new_values, new_labels

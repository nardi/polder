from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from functools import reduce
from types import ModuleType
from typing import TYPE_CHECKING, Any, Generic, NamedTuple, TypeAlias, cast

import narwhals as nw
import narwhals.typing as nwt

if TYPE_CHECKING:
    from polder.lazy.array import (
        ExternalFrameType,
        LazyFrameLabeledArray,
    )

_NO_FILL = object()

AxisLabelsSpecifier: TypeAlias = Sequence[str]
AxisLabelsToPivot: TypeAlias = AxisLabelsSpecifier | Sequence[AxisLabelsSpecifier]
AxesSlice = tuple[int, int]


class ArrayAttributes(Generic[nwt.FrameT], NamedTuple):
    """Convenience class to hold the data needed to construct a LazyFrameLabeledArray."""

    indexed_labels: tuple[nwt.FrameT | None, ...]
    values: nwt.FrameT
    shape: nwt.FrameT
    n_dims: int
    frame_ns: ModuleType


def pivot(
    arr: LazyFrameLabeledArray[ExternalFrameType, nwt.FrameT],
    /,
    *,
    axis_labels_to_pivot: Mapping[int, AxisLabelsToPivot],
    fill_value: Any = _NO_FILL,
) -> LazyFrameLabeledArray[ExternalFrameType, nwt.FrameT]:
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
    arr_attrs = ArrayAttributes(
        arr._indexed_labels, arr._values, arr._shape, arr._n_dims, arr._frame_ns
    )

    # Pivot each axis one at a time, processing in descending order so earlier
    # pivots do not affect axis indices of later pivots.
    for axis in sorted(axis_labels_to_pivot.keys(), reverse=True):
        arr_attrs = _pivot_single_axis(
            arr_attrs, axis, axis_labels_to_pivot[axis], fill_value
        )

    return type(arr)(
        arr_attrs.indexed_labels,
        arr_attrs.values,
        arr_attrs.shape,
        arr_attrs.n_dims,
        arr_attrs.frame_ns,
    )


def _pivot_single_axis(
    arr: ArrayAttributes[nwt.FrameT],
    axis: int,
    pivot_spec: AxisLabelsToPivot,
    fill_value: Any,
) -> ArrayAttributes[nwt.FrameT]:
    """Pivot a single axis, potentially creating one or more new axes."""
    pivot_groups = _normalize_pivot_spec(pivot_spec)

    # Handle each pivot group in reverse order (so that unspecified columns are
    # implicitly first in the axis order).
    for pivot_cols in reversed(pivot_groups):
        arr = _pivot_single_group(arr, axis, pivot_cols, fill_value)

    return arr


def _normalize_pivot_spec(spec: AxisLabelsToPivot) -> list[list[str]]:
    """Normalize pivot specification to a list of lists of column names."""
    # If spec is a sequence of strings, it's a single group.
    if all(isinstance(s, str) for s in spec):
        spec = cast(list[str], list(spec))
        return [spec]
    # Otherwise, it's a sequence of groups.
    return [list(group) for group in spec]


def _pivot_single_group(
    arr: ArrayAttributes[nwt.FrameT],
    axis: int,
    pivot_cols: list[str],
    fill_value: Any,
) -> ArrayAttributes[nwt.FrameT]:
    labels_frame = arr.indexed_labels[axis]
    if labels_frame is None:
        raise ValueError(f"Cannot pivot unlabeled axis {axis}")

    # Determine which columns to keep vs which to pivot.
    all_cols = labels_frame.collect_schema().names()
    keep_cols = [c for c in all_cols if c not in ("__index", *pivot_cols)]

    # If there are no columns to keep, that means we are already done.
    if not keep_cols:
        return arr

    # Determine unique combinations of keep columns (these become the new axis
    # labels).
    keep_labels = cast(
        nwt.FrameT,
        labels_frame
        .select(nw.col("__index").alias("__orig_index"), *keep_cols)
        .unique(keep_cols)
        .with_row_index("__index", order_by="__orig_index")
        .drop("__orig_index"),
    )

    # Determine unique combinations of pivot columns (these become the new axis
    # labels to insert).
    pivot_labels = cast(
        nwt.FrameT,
        labels_frame
        .select(nw.col("__index").alias("__orig_index"), *pivot_cols)
        .unique(pivot_cols)
        .with_row_index("__index", order_by="__orig_index")
        .drop("__orig_index"),
    )

    # Insert the new label frames in the right location.
    indexed_labels = (
        *arr.indexed_labels[:axis],
        keep_labels,
        pivot_labels,
        *arr.indexed_labels[axis + 1 :],
    )

    n_dims = arr.n_dims + 1

    # Deduce the new shape from the new label frames.
    axis_dtype, size_dtype = arr.shape.select("axis", "size").collect_schema().dtypes()
    shape = nw.concat([
        arr.shape.filter(nw.col("axis") < axis),
        keep_labels.select(
            nw.lit(axis, axis_dtype).alias("axis"),
            nw.col("__index").count().cast(size_dtype).alias("size"),
        ),
        pivot_labels.select(
            nw.lit(axis + 1, axis_dtype).alias("axis"),
            nw.col("__index").count().cast(size_dtype).alias("size"),
        ),
        arr.shape.filter(nw.col("axis") > axis).with_columns(nw.col("axis") + 1),
    ])

    # Now, to transform the values we need to find the full set of indices for
    # all axes. This will include new (keep, pivot) label combinations that
    # previously didn't exist. Onto these combinations, we want to "broadcast"
    # all other axes. Then, depending on `fill_value`, we will either fill or
    # not fill missing values.

    # First, relabel the current indices in `values`. We have to do two things:
    #   1. Move the later axes out of the way: we add an axis at `axis + 1`, so
    #      any axes from that point will have their index increased by 1.
    values = arr.values.rename({
        f"__index{i}": f"__index{i + 1}" for i in range(axis + 1, arr.n_dims)
    })
    #   2. Join the new axis onto the values frame, using the labels of the
    #      pivoted axis as a mapping.
    pivoted_axis_idx_mapping = (
        labels_frame
        .rename({"__index": "__pivoted_index"})
        .join(
            keep_labels.rename({"__index": f"__index{axis}"}),  # type: ignore
            on=keep_cols,
            how="left",
        )
        .join(
            pivot_labels.rename({"__index": f"__index{axis + 1}"}),  # type: ignore
            on=pivot_cols,
            how="left",
        )
        .select("__pivoted_index", f"__index{axis}", f"__index{axis + 1}")
    )
    values = (
        values
        .rename({f"__index{axis}": "__pivoted_index"})
        .join(
            pivoted_axis_idx_mapping,  # type: ignore
            on="__pivoted_index",
            how="left",
        )
        .drop("__pivoted_index")
    )

    # If we have no fill value, we can stop here. Since we are not filling
    # anything, we will notice the missing values eventually and then panic, so
    # no need to add them into the `values` frame.
    if fill_value is not _NO_FILL:
        # Create a frame containing value indices for all label combinations.
        index_columns = [f"__index{i}" for i in range(n_dims)]
        all_labels_idx = reduce(
            lambda a, b: a.join(b, how="cross"),  # type: ignore
            [
                frame.select(nw.col("__index").alias(col))
                if frame is not None
                else nw.from_dict({col: [0]}, native_namespace=arr.frame_ns)
                for col, frame in zip(index_columns, indexed_labels)
            ],
        )

        # Anti-join the values onto these in order to find non-existent combinations.
        values_schema = values.collect_schema()
        filled_missing_values = all_labels_idx.join(
            values,  # type: ignore
            on=index_columns,
            how="anti",
        ).with_columns(nw.lit(fill_value).cast(values_schema["value"]).alias("value"))

        # Combine the existing and filled values.
        values = nw.concat([
            values.select(*index_columns, "value"),
            filled_missing_values.select(*index_columns, "value"),
        ])

    # A cast to satisfy the type checker.
    # TODO: can this be avoided?
    values = cast(nwt.FrameT, values)

    return ArrayAttributes(indexed_labels, values, shape, n_dims, arr.frame_ns)


def unpivot(
    arr: LazyFrameLabeledArray[ExternalFrameType, nwt.FrameT],
    /,
    *,
    axes_to_merge: Sequence[tuple[int, int]],
) -> LazyFrameLabeledArray[ExternalFrameType, nwt.FrameT]:
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

    indexed_labels = arr._indexed_labels
    values = arr._values
    shape = arr._shape
    shape_schema = shape.collect_schema()
    n_dims = arr._n_dims

    for first_axis, last_axis in reversed(axes_to_merge):
        axes_labels = indexed_labels[first_axis : last_axis + 1]
        index_columns = [f"__index{i}" for i in range(first_axis, last_axis + 1)]

        unlabeled_count = sum(1 for axis_labels in axes_labels if axis_labels is None)
        if unlabeled_count:
            if unlabeled_count != len(axes_labels):
                raise ValueError("Cannot merge unlabeled axes with labeled ones.")

            # All axes are unlabeled, so we can simply merge them together into
            # one unlabeled axis.
            merged_axis_labels = None

            value_idx_mapping = nw.from_dict(
                {col: [0] for col in [*index_columns, "__index"]},
                native_namespace=arr._frame_ns,
            )
        else:
            axes_labels = cast(tuple[nwt.FrameT, ...], axes_labels)

            # Create the merged labels as a cross product of the axes.
            merged_axis_labels = reduce(
                lambda a, b: a.join(b, how="cross"),  # type: ignore
                [
                    frame.select(
                        nw.col("__index").alias(idx_col), nw.exclude("__index")
                    )
                    for idx_col, frame in zip(index_columns, axes_labels)
                ],
            ).with_row_index("__index", order_by=index_columns)

            value_idx_mapping = merged_axis_labels.select(*index_columns, "__index")

            merged_axis_labels = merged_axis_labels.drop(index_columns)

        # Replace the existing labels by the merged ones.
        indexed_labels = (
            *indexed_labels[:first_axis],
            merged_axis_labels,
            *indexed_labels[last_axis + 1 :],
        )

        # Map the merged indices in the values frame.
        values = (
            values
            .join(value_idx_mapping, on=index_columns, how="left")  # type: ignore
            .drop(index_columns)
            .rename({"__index": f"__index{first_axis}"})
        )
        # Shift all later axes down.
        axes_removed = last_axis - first_axis
        values = values.rename({
            f"__index{i}": f"__index{i - axes_removed}"
            for i in range(last_axis + 1, n_dims)
        })

        # Determine the new shape frame. The shape for the merged axis will be
        # the product of all existing axes.
        shape = nw.concat([
            shape.filter(nw.col("axis") < first_axis),
            shape
            .filter(nw.col("axis").is_between(first_axis, last_axis))
            .with_columns(nw.col("size").cum_prod().over(order_by="axis").max())
            .with_columns(nw.lit(first_axis, dtype=shape_schema["axis"]).alias("axis"))
            .unique(),
            shape.filter(nw.col("axis") > last_axis).with_columns(
                nw.col("axis") - axes_removed
            ),
        ])

        n_dims -= axes_removed

    return type(arr)(indexed_labels, values, shape, n_dims, arr._frame_ns)

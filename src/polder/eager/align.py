from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any, Generic, NamedTuple, TypeAlias, TypeVar

import narwhals as nw
import numpy as np
from immutabledict import immutabledict
from narwhals.typing import DataFrameT

from polder.eager._narwhals_df_equals import narwhals_df_equals
from polder.protocols.array import AnyDataFrame

if TYPE_CHECKING:
    from polder.eager.array import SomeEagerFrameLabeledArray


class AlignmentResults(Generic[DataFrameT], NamedTuple):
    labels: DataFrameT | None
    value_indices: tuple[slice | np.ndarray, ...]


T = TypeVar("T")

OrderedSet: TypeAlias = immutabledict[T, None]


def orderedset(items: Iterable[T]) -> OrderedSet[T]:
    return immutabledict.fromkeys(items)


def _determine_alignment_cols(label_frame: nw.DataFrame[Any]) -> OrderedSet[str]:
    (has_multiple_unique_values,) = label_frame.select(
        nw.all().n_unique() > 1
    ).iter_rows(named=True)
    return orderedset(
        col for col in label_frame.columns if has_multiple_unique_values[col]
    )


def _align_labels(label_frames: Sequence[DataFrameT]) -> AlignmentResults[DataFrameT]:
    # If there is only one frame, nothing to do.
    if len(label_frames) == 1:
        return AlignmentResults(label_frames[0], (slice(None),))

    # Determine alignment columns per label frame.
    all_alignment_cols = frozenset(map(_determine_alignment_cols, label_frames))

    if len(all_alignment_cols) > 1:
        raise Exception(
            f"Cannot align labels, because there are unmatched columns between them: {all_alignment_cols}"
        )

    (alignment_cols,) = all_alignment_cols
    alignment_cols = list(alignment_cols)

    if alignment_cols == []:
        # No alignment columns, so remove the labels.
        return AlignmentResults(None, (slice(None),) * len(label_frames))

    # Join all frames onto the first to determine how their values should be reordered.
    aligned_labels = label_frames[0].select(alignment_cols)
    value_indices = (
        slice(None),
        *[
            aligned_labels.join(
                frame.with_row_index("__value_indices"),
                on=alignment_cols,
                how="left",
            )["__value_indices"].to_numpy()
            for frame in label_frames[1:]
        ],
    )

    return AlignmentResults(aligned_labels, value_indices)


def _subset_labels(label_frames: Sequence[DataFrameT]) -> DataFrameT | None:
    """Checks if frames are already aligned, and if so, subset the columns to the alignment
    columns."""
    # If there is only one frame, nothing to do.
    if len(label_frames) == 1:
        return label_frames[0]

    # Determine alignment columns per label frame.
    all_alignment_cols = frozenset(map(_determine_alignment_cols, label_frames))

    if len(all_alignment_cols) > 1:
        raise Exception(
            f"Cannot align labels, because there are unmatched columns between them: {all_alignment_cols}"
        )

    (alignment_cols,) = all_alignment_cols
    alignment_cols = list(alignment_cols)

    if alignment_cols == []:
        # No alignment columns, so remove the labels.
        return None

    # Check that all frames are equal after subsetting.
    aligned_labels = label_frames[0].select(alignment_cols)
    unaligned_frames: list[DataFrameT] = []
    for frame in label_frames[1:]:
        subset_frame = frame.select(alignment_cols)
        if not narwhals_df_equals(aligned_labels, subset_frame):
            unaligned_frames.append(subset_frame)
    if unaligned_frames:
        raise Exception(
            "Cannot combine arrays with unaligned labels:\n"
            + "\n".join(str(frame) for frame in [aligned_labels, *unaligned_frames])
        )

    return aligned_labels


AxisNumbers: TypeAlias = tuple[int, ...]
"""A set of axes to align over the provided arrays."""


def align(
    *arrays: SomeEagerFrameLabeledArray,
    axes: Iterable[AxisNumbers] | None = None,
    drop_single_valued_labels: bool = True,
    check_only: bool = False,
) -> tuple[SomeEagerFrameLabeledArray, ...]:
    """Aligns a number of frame-labeled arrays along all axes. The alignment rules are as follows:

    1. All arrays must have the same number of axes.
    2. Single-valued/unlabeled axes are already aligned, since they will be broadcasted.
    3. Scalars are already aligned, since they will be broadcasted.
    4. All label frames for a single axis will be aligned by reordering, so they must have the same
       length.
    5. The columns in a label frame with more than 1 unique value are its alignment columns.
       Single-value columns will be ignored.
    6. All label frames for a single axis must have the same alignment columns.
    7. After alignment, all label frames for each axis will be identical, unless they are `None` or
       have only a single row.
    8. Label frames with a single row don't need alignment (they'll be broadcasted). By default they
       are dropped (set to None), but this can be avoided by using
       `drop_single_valued_labels=False`.

    Args:
        axes: Optional specification of axes to align. If None, aligns all axes.
        drop_single_valued_labels: If True, removes labels for axes that have only one row
            (since these will be broadcasted anyway). Defaults to True.
        check_only: If True, only check if the arrays are aligned, and if so return them. Will not
        modify anything.
    """
    if not arrays:
        return ()

    # Get the array namespace corresponding to the first array.
    xp = arrays[0].array_namespace

    # Extract labels and values for all arrays.
    all_labels = [list(arr.labels()) for arr in arrays]
    all_values = [arr.values() for arr in arrays]

    # If no axes are provided, align on all axes.
    if axes is None:
        n_axes_per_array = set(len(values.shape) for values in all_values)

        if len(n_axes_per_array) != 1:
            raise Exception(
                "Cannot determine axes to align arrays on, as no axes were provided and arrays "
                "are not identically shaped."
            )

        (n_axes,) = n_axes_per_array
        n_arrays = len(arrays)
        axes = tuple(((i,) * n_arrays) for i in range(n_axes))

    axes = tuple(axes)

    # Align every set of axes one-by-one.
    for axis in axes:
        axis_lengths = set(
            values.shape[i] for i, values in zip(axis, all_values, strict=True)
        )

        # Check that the axis lengths are all the same, with single-valued (broadcastable) axes as
        # an exception.
        if len(axis_lengths | {1}) not in (1, 2):
            raise Exception(
                f"Cannot align arrays on axes with incompatible lengths: {axis_lengths}."
            )

        # Collect alignable (multi-row) label frames, and their associated axis positions.
        alignable_label_frames: list[AnyDataFrame] = []
        # Contains (array_idx, axis_idx) for each multi-row label frame.
        alignable_label_frame_positions: list[tuple[int, int]] = []

        for array_idx, (axis_idx, arr_labels) in enumerate(
            zip(axis, all_labels, strict=True)
        ):
            label_frame = arr_labels[axis_idx]
            if label_frame is None:
                continue
            elif len(label_frame) == 1:
                if drop_single_valued_labels:
                    all_labels[array_idx][axis_idx] = None
            else:
                alignable_label_frames.append(label_frame)
                alignable_label_frame_positions.append((array_idx, axis_idx))

        if alignable_label_frames:
            if check_only:
                # Check that the label frames are aligned, and possibly subset the columns.
                aligned_labels_for_axis = _subset_labels(tuple(alignable_label_frames))

                # Replace the original labels with the subsetted ones.
                for array_idx, axis_idx in alignable_label_frame_positions:
                    all_labels[array_idx][axis_idx] = aligned_labels_for_axis
            else:
                # Align the label frames and get reindexing information.
                aligned_labels_for_axis, value_indices = _align_labels(
                    tuple(alignable_label_frames)
                )

                # Replace the original labels with the aligned ones and reindex the arrays.
                for (array_idx, axis_idx), value_idx in zip(
                    alignable_label_frame_positions, value_indices, strict=True
                ):
                    # Convert value indices to the appropriate array type before
                    # indexing.
                    if isinstance(value_idx, np.ndarray):
                        value_idx = xp.asarray(value_idx)

                    all_labels[array_idx][axis_idx] = aligned_labels_for_axis
                    all_values[array_idx] = all_values[array_idx][
                        (slice(None),) * axis_idx + (value_idx,)
                    ]

    return tuple(
        original_array.create(array_labels, array_values)
        for original_array, array_labels, array_values in zip(
            arrays, all_labels, all_values, strict=True
        )
    )

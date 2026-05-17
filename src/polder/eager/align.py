from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Generic, NamedTuple, TypeAlias, TypeVar

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


def _align_labels(label_frames: Sequence[DataFrameT]) -> AlignmentResults[DataFrameT]:
    # If there is only one frame, nothing to do.
    if len(label_frames) == 1:
        return AlignmentResults(label_frames[0], (slice(None),))

    # Determine columns per label frame.
    all_columns = frozenset(orderedset(frame.columns) for frame in label_frames)

    if len(all_columns) > 1:
        raise Exception(
            f"Cannot align labels, because they have different columns: {all_columns}"
        )

    (columns,) = all_columns
    columns = list(columns)

    if not columns:
        # No columns, so no need to do any alignment.
        return AlignmentResults(label_frames[0], (slice(None),) * len(label_frames))

    # Join all frames onto the first to determine how their values should be reordered.
    aligned_labels = label_frames[0].select(columns)
    value_indices = (
        slice(None),
        *[
            aligned_labels.join(
                frame.with_row_index("__value_indices"),
                on=columns,
                how="left",
            )["__value_indices"].to_numpy()
            for frame in label_frames[1:]
        ],
    )

    if any(np.isnan(value_idx).any() for value_idx in value_indices[1:]):
        # TODO: give some feedback here on which labels are differing.
        raise KeyError("Cannot align arrays with differing labels")

    return AlignmentResults(aligned_labels, value_indices)


def _check_label_alignment(label_frames: Sequence[DataFrameT]) -> None:
    """Checks if label frames are already aligned."""
    # If there is only one frame, nothing to do.
    if len(label_frames) == 1:
        return

    # Determine columns per label frame.
    all_columns = frozenset(orderedset(frame.columns) for frame in label_frames)

    if len(all_columns) > 1:
        raise Exception(
            f"Cannot align labels, because they have different columns: {all_columns}"
        )

    (columns,) = all_columns
    columns = list(columns)

    # Check that all frames are equal after reordering columns.
    aligned_labels = label_frames[0].select(columns)
    unaligned_frames: list[DataFrameT] = []
    for frame in label_frames[1:]:
        other_labels = frame.select(columns)
        if not narwhals_df_equals(aligned_labels, other_labels):
            unaligned_frames.append(other_labels)
    if unaligned_frames:
        raise Exception(
            "Cannot combine arrays with unaligned labels:\n"
            + "\n".join(str(frame) for frame in [aligned_labels, *unaligned_frames])
        )


AxisNumbers: TypeAlias = tuple[int, ...]
"""A set of axes to align over the provided arrays."""


def align(
    *arrays: SomeEagerFrameLabeledArray,
    axes: Iterable[AxisNumbers] | None = None,
    check_only: bool = False,
) -> tuple[SomeEagerFrameLabeledArray, ...]:
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
                "Cannot determine axes to align arrays on, as no axes "
                "were specified and arrays have different dimensionality."
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

        # Check that the axis lengths are all the same, with single-valued
        # (broadcastable) axes as an exception.
        if len(axis_lengths | {1}) not in (1, 2):
            raise Exception(
                f"Cannot align arrays on axes with incompatible lengths: {axis_lengths}."
            )

        # Collect alignable (multi-row) label frames, and their associated axis
        # positions.
        alignable_label_frames: list[AnyDataFrame] = []
        # Contains (array_idx, axis_idx) for each multi-row label frame.
        alignable_label_frame_positions: list[tuple[int, int]] = []

        for array_idx, (axis_idx, arr_labels) in enumerate(
            zip(axis, all_labels, strict=True)
        ):
            label_frame = arr_labels[axis_idx]
            if label_frame is None:
                continue

            alignable_label_frames.append(label_frame)
            alignable_label_frame_positions.append((array_idx, axis_idx))

        if alignable_label_frames:
            if check_only:
                # Check that the label frames are aligned.
                _check_label_alignment(tuple(alignable_label_frames))
            else:
                # Align the label frames and get reindexing information.
                aligned_labels_for_axis, value_indices = _align_labels(
                    tuple(alignable_label_frames)
                )

                # Replace the original labels with the aligned ones and reindex
                # the arrays.
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
        original_array.from_values_and_labels(array_values, array_labels)
        for original_array, array_labels, array_values in zip(
            arrays, all_labels, all_values, strict=True
        )
    )

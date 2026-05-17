from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Generic, NamedTuple, TypeAlias

import narwhals as nw
import narwhals.typing as nwt

from polder.utils.orderedset import orderedset

if TYPE_CHECKING:
    from polder.lazy.array import SomeLazyFrameLabeledArray


AxisNumbers: TypeAlias = tuple[int, ...]
"""A set of axes to align over the provided arrays."""


class AlignmentResults(Generic[nwt.LazyFrameT], NamedTuple):
    labels: tuple[nwt.LazyFrameT, ...]
    """Contains the labels for each axis after alignment."""
    value_index_mappings: tuple[nwt.LazyFrameT | None, ...]
    """Contains a two-column frame (`__old_index` and `__index`) that shows how
    the values should be reordered for each axis. `None` means no reordering is
    necessary."""


def _align_labels(
    label_frames: Sequence[nwt.LazyFrameT],
) -> AlignmentResults[nwt.LazyFrameT]:
    # If there is only one frame, nothing to do.
    if len(label_frames) == 1:
        return AlignmentResults(tuple(label_frames[:1]), (None,))

    # Determine label columns per label frame.
    all_label_columns = frozenset(
        orderedset(col for col in frame.collect_schema().names() if col != "__index")
        for frame in label_frames
    )

    if len(all_label_columns) > 1:
        raise Exception(
            f"Cannot align labels, because they have different columns: {all_label_columns}"
        )

    (label_columns,) = all_label_columns
    label_columns = list(label_columns)

    if not label_columns:
        # No columns, so no need to do any alignment.
        return AlignmentResults(tuple(label_frames), (None,) * len(label_frames))

    # The set of aligned labels is given by the first frame.
    labels = [label_frames[0].select("__index", *label_columns)]
    value_index_mappings: list[nwt.LazyFrameT | None] = [None]
    # Then, each subsequent frame can deduce an index mapping by joining onto
    # this frame.
    reference_labels = labels[0]
    for other_label_frame in label_frames[1:]:
        new_label_frame = other_label_frame.rename({"__index": "__old_index"}).join(
            reference_labels, on=label_columns, how="left"
        )
        labels.append(new_label_frame.select("__index", *label_columns))
        value_index_mappings.append(new_label_frame.select("__old_index", "__index"))

    return AlignmentResults(tuple(labels), tuple(value_index_mappings))


def align(
    *arrays: SomeLazyFrameLabeledArray,
    axes: Iterable[AxisNumbers] | None = None,
    check_only: bool = False,
) -> tuple[SomeLazyFrameLabeledArray, ...]:
    """Alignment for lazy arrays. Note that for lazy arrays, there is no way to
    check the alignment other than to attempt it and observe it failing when
    executed, so `check_only` has no effect and alignment is always
    performed."""
    if not arrays:
        return ()

    # Extract labels and values for all arrays.
    all_labels = [list(arr._indexed_labels) for arr in arrays]
    all_values = [arr._values for arr in arrays]
    all_shapes = [arr._shape for arr in arrays]

    # If no axes are provided, align on all axes.
    if axes is None:
        n_axes_per_array = set(arr._n_dims for arr in arrays)

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
        # Collect alignable label frames.
        alignable_label_frames: list[nw.LazyFrame] = []
        # Contains (array_idx, axis_idx) for each alignable label frame.
        alignable_label_frame_positions: list[tuple[int, int]] = []

        for array_idx, (axis_idx, arr_labels) in enumerate(
            zip(axis, all_labels, strict=True)
        ):
            label_frame = arr_labels[axis_idx]
            if label_frame is None:
                continue
            else:
                alignable_label_frames.append(label_frame)
                alignable_label_frame_positions.append((array_idx, axis_idx))

        if alignable_label_frames:
            # Align the label frames and get reindexing information.
            aligned_labels, value_index_mappings = _align_labels(
                tuple(alignable_label_frames)
            )

            # Replace the original labels with the aligned ones and reindex
            # the values.
            for aligned_labels_for_axis, (
                array_idx,
                axis_idx,
            ), value_idx_mapping in zip(
                aligned_labels,
                alignable_label_frame_positions,
                value_index_mappings,
                strict=True,
            ):
                all_labels[array_idx][axis_idx] = aligned_labels_for_axis

                if value_idx_mapping is not None:
                    value_idx_col = f"__index{axis_idx}"
                    all_values[array_idx] = (
                        all_values[array_idx]
                        .rename({value_idx_col: "__old_index"})
                        .join(
                            value_idx_mapping.rename({"__index": value_idx_col}),
                            on="__old_index",
                            how="left",
                        )
                        .drop("__old_index")
                    )

            # Unify the shapes of the aligned axes. These should all be
            # identical. We use the shape of the first array as reference, and
            # check the rest against it.
            (first_array_idx, first_axis_idx), *_ = alignable_label_frame_positions
            reference_shape = all_shapes[first_array_idx].filter(axis=first_axis_idx)
            reference_shape_schema = reference_shape.collect_schema()
            for array_idx, axis_idx in alignable_label_frame_positions[1:]:
                all_shapes[array_idx] = nw.concat(
                    [
                        all_shapes[array_idx],
                        reference_shape.with_columns(
                            axis=nw.lit(axis_idx, reference_shape_schema["axis"])
                        ),
                    ],
                    how="vertical",
                ).unique()

    return tuple(
        type(original_array)(
            tuple(array_labels),
            array_values,
            array_shape,
            original_array._n_dims,
            original_array._frame_ns,
        )
        for original_array, array_labels, array_values, array_shape in zip(
            arrays, all_labels, all_values, all_shapes, strict=True
        )
    )

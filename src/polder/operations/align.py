from collections.abc import Iterable
from typing import TypeAlias, cast

from polder.eager.align import align as eager_align
from polder.eager.array import EagerFrameLabeledArray
from polder.protocols.array import SomeFrameLabeledArray

AxisNumbers: TypeAlias = tuple[int, ...]
"""A set of axes to align over the provided arrays."""


def align(
    *arrays: SomeFrameLabeledArray,
    axes: Iterable[AxisNumbers] | None = None,
    drop_single_valued_labels: bool = True,
    check_only: bool = False,
) -> tuple[SomeFrameLabeledArray, ...]:
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
    if all(isinstance(arr, EagerFrameLabeledArray) for arr in arrays):
        eager_arrays = cast(tuple[EagerFrameLabeledArray, ...], arrays)
        return cast(
            tuple[SomeFrameLabeledArray, ...],
            eager_align(
                *eager_arrays,
                axes=axes,
                drop_single_valued_labels=drop_single_valued_labels,
                check_only=check_only,
            ),
        )

    raise NotImplementedError()

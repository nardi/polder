from collections.abc import Iterable
from typing import TypeAlias, cast

from polder.eager.align import align as eager_align
from polder.eager.array import EagerFrameLabeledArray
from polder.lazy.align import align as lazy_align
from polder.lazy.array import LazyFrameLabeledArray
from polder.protocols.array import SomeFrameLabeledArray

AxisNumbers: TypeAlias = tuple[int, ...]
"""A set of axes to align over the provided arrays."""


def align(
    *arrays: SomeFrameLabeledArray,
    axes: Iterable[AxisNumbers] | None = None,
    check_only: bool = False,
) -> tuple[SomeFrameLabeledArray, ...]:
    """Aligns a number of frame-labeled arrays along all axes. The alignment rules are as follows:

    1. All arrays must have the same number of axes.
    2. Unlabeled (size-1) axes are already aligned, since they will be broadcasted.
    3. Scalars are already aligned, since they will be broadcasted.
    4. All label frames for a single axis will be aligned by reordering, so they must have the same
       length.
    5. All label frames for a single axis must have the same columns.
    6. After alignment, all label frames for each axis will be identical, unless they are `None`.

    Args:
        axes: Optional specification of axes to align. If None, aligns all axes.
        check_only: If True, only check whether the arrays are aligned, and if so return
            them unchanged. When True, nothing is reordered.

    Returns:
        The input arrays, reordered so that their labels are aligned along every axis.
    """
    if all(isinstance(arr, EagerFrameLabeledArray) for arr in arrays):
        eager_arrays = cast(tuple[EagerFrameLabeledArray, ...], arrays)
        return cast(
            tuple[SomeFrameLabeledArray, ...],
            eager_align(
                *eager_arrays,
                axes=axes,
                check_only=check_only,
            ),
        )

    if all(isinstance(arr, LazyFrameLabeledArray) for arr in arrays):
        lazy_arrays = cast(tuple[LazyFrameLabeledArray, ...], arrays)
        return cast(
            tuple[SomeFrameLabeledArray, ...],
            lazy_align(
                *lazy_arrays,
                axes=axes,
                check_only=check_only,
            ),
        )

    raise NotImplementedError("`align` is not implemented for this array type")

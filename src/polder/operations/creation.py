from collections.abc import Iterable
from typing import overload

import narwhals as nw
import numpy as np
from narwhals.typing import IntoDataFrameT
from optype.numpy import Array

from polder.eager.array import EagerFrameLabeledArray
from polder.protocols.array import AnyArray, FrameLabeledArray


@overload
def from_values_and_labels(
    values: Array, labels: Iterable[IntoDataFrameT]
) -> EagerFrameLabeledArray[nw.DataFrame[IntoDataFrameT]]: ...


@overload
def from_values_and_labels(
    values: AnyArray, labels: Iterable[IntoDataFrameT]
) -> FrameLabeledArray[nw.DataFrame[IntoDataFrameT], AnyArray]: ...


def from_values_and_labels(
    values: AnyArray, labels: Iterable[IntoDataFrameT]
) -> FrameLabeledArray[nw.DataFrame[IntoDataFrameT], AnyArray]:
    label_dfs = tuple(map(nw.from_native, labels))
    if isinstance(values, np.ndarray):
        return EagerFrameLabeledArray(label_dfs, values)
    raise NotImplementedError()

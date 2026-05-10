from collections.abc import Iterable
from typing import overload

import narwhals as nw
import numpy as np
from narwhals.typing import IntoDataFrameT

from polder.eager.array import EagerFrameLabeledArray
from polder.eager.value_array import SomeValueArray
from polder.protocols.array import AnyArray, FrameLabeledArray


@overload
def from_values_and_labels(
    values: SomeValueArray, labels: Iterable[IntoDataFrameT]
) -> EagerFrameLabeledArray[nw.DataFrame[IntoDataFrameT], SomeValueArray]: ...


# We add an overload for np.ndarray specifically because the concrete class
# doesn't validate against Array from `types-array-api`. This way we can do a
# typecast here and let both library and user code be typed correctly.
@overload
def from_values_and_labels(
    values: np.ndarray, labels: Iterable[IntoDataFrameT]
) -> FrameLabeledArray[nw.DataFrame[IntoDataFrameT], np.ndarray]: ...


def from_values_and_labels(
    values: AnyArray, labels: Iterable[IntoDataFrameT]
) -> FrameLabeledArray[nw.DataFrame[IntoDataFrameT], AnyArray]:
    label_dfs = tuple(map(nw.from_native, labels))
    if isinstance(values, np.ndarray):
        # np.ndarray doesn't type as Array for some reason.
        return EagerFrameLabeledArray(label_dfs, values)  # type: ignore
    raise NotImplementedError()

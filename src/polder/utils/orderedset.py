from collections.abc import Iterable
from typing import TypeAlias, TypeVar

from immutabledict import immutabledict

T = TypeVar("T")

OrderedSet: TypeAlias = immutabledict[T, None]


def orderedset(items: Iterable[T]) -> OrderedSet[T]:
    return immutabledict.fromkeys(items)

from collections.abc import Callable
from typing import Concatenate, Generic, ParamSpec, TypeVar

SelfParam = TypeVar("SelfParam")
OtherParams = ParamSpec("OtherParams")
Return = TypeVar("Return")


class BoundIndexerMethod(Generic[SelfParam, OtherParams, Return]):
    __slots__ = ("_func", "_instance")

    _func: Callable[Concatenate[SelfParam, OtherParams], Return]
    _instance: SelfParam

    def __init__(
        self,
        func: Callable[Concatenate[SelfParam, OtherParams], Return],
        instance: SelfParam,
    ):
        self._func = func
        self._instance = instance

    def __call__(self, *args: OtherParams.args, **kwargs: OtherParams.kwargs) -> Return:
        return self._func(self._instance, *args, **kwargs)

    def __getitem__(self, subscript, /) -> Return:
        # TODO: type this function properly.
        if isinstance(subscript, tuple):
            return self._func(self._instance, *subscript)  # type: ignore
        return self._func(self._instance, subscript)  # type: ignore


class IndexerMethod(Generic[SelfParam, OtherParams, Return]):
    __slots__ = ("_func",)

    _func: Callable[Concatenate[SelfParam, OtherParams], Return]

    def __init__(self, func: Callable[Concatenate[SelfParam, OtherParams], Return]):
        self._func = func

    def __get__(
        self, obj: SelfParam, objtype: type[SelfParam] | None = None
    ) -> BoundIndexerMethod[SelfParam, OtherParams, Return]:
        return BoundIndexerMethod(self._func, obj)


def indexermethod(
    f: Callable[Concatenate[SelfParam, OtherParams], Return],
) -> IndexerMethod[SelfParam, OtherParams, Return]:
    """A decorator that can be applied to a method to wrap it into a
    `IndexerMethod` object, that allows calling the method regularly, but also
    with __getitem__ syntax (like `f[1, :, ...]`)."""

    return IndexerMethod(f)

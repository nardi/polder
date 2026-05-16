from typing import Generic, Protocol, TypeVar

InstanceType = TypeVar("InstanceType", contravariant=True)
ValueType = TypeVar("ValueType", covariant=True)


class Descriptor(Generic[InstanceType, ValueType], Protocol):
    def __get__(
        self, obj: InstanceType, objtype: type[InstanceType] | None = None
    ) -> ValueType: ...

"""Global configuration settings for polder."""

from contextlib import _GeneratorContextManager, contextmanager
from contextvars import ContextVar
from typing import Generator, overload

_auto_align: ContextVar[bool] = ContextVar("auto_align", default=True)
"""When performing operations with eager arrays, alignment is performed
automatically if this setting is true."""

_use_eager_evaluation_for_lazy_arrays: ContextVar[bool] = ContextVar(
    "use_eager_evaluation_for_lazy_arrays", default=False
)
"""Use DataFrames instead of LazyFrames for lazy arrays. This can be useful for
testing purposes, because errors will surface more quickly and closer to where
they originate."""


@overload
def auto_align() -> bool: ...


@overload
def auto_align(enable: bool) -> _GeneratorContextManager[None, None, None]: ...


@overload
def use_eager_evaluation_for_lazy_arrays() -> bool: ...


@overload
def use_eager_evaluation_for_lazy_arrays(
    enable: bool,
) -> _GeneratorContextManager[None, None, None]: ...


def auto_align(
    enable: bool | None = None,
) -> bool | _GeneratorContextManager[None, None, None]:
    """When performing operations with eager arrays, alignment is performed
    automatically if this setting is true.

    When called without arguments, returns the current auto_align value.
    When called with an argument, returns a context manager during which the
    setting has the provided value.

    Args:
        enable: Whether to enable or disable automatic alignment of arrays in binary
            operations. When False, alignment is only checked but not performed. When
            None (the default), the current value is returned instead.

    Returns:
        The current setting as a bool when enable is None, otherwise a context manager
        during which the setting has the provided value.

    Example:
        ```python
        import polder as pld

        # Get current setting
        current = pld.config.auto_align()

        # Disable auto-alignment temporarily
        with pld.config.auto_align(False):
            result = arr1 + arr2  # Only checks if alignment is needed
        ```
    """
    if enable is None:
        # Get current value
        return _auto_align.get()
    else:
        # Return context manager
        @contextmanager
        def _context_manager() -> Generator[None, None, None]:
            token = _auto_align.set(enable)
            try:
                yield
            finally:
                _auto_align.reset(token)

        return _context_manager()


def use_eager_evaluation_for_lazy_arrays(
    enable: bool | None = None,
) -> bool | _GeneratorContextManager[None, None, None]:
    """Use DataFrames instead of LazyFrames for lazy arrays. This can be useful for
    testing purposes, because errors will surface more quickly and closer to where
    they originate.

    Note that some errors will still only surface lazily. For example an
    "invalid shape" error may only arise when the shape is extracted, not on the
    operation that produces the invalid shape.

    When called without arguments, returns the current setting value.
    When called with an argument, returns a context manager during which the
    setting has the provided value.

    Args:
        enable: Whether to enable eager evaluation for lazy arrays. When None (the
            default), the current value is returned instead.

    Returns:
        The current setting as a bool when enable is None, otherwise a context manager
        during which the setting has the provided value.

    Example:
        ```python
        import polder as pld

        with pld.config.use_eager_evaluation_for_lazy_arrays(True):
            lazy_array = pld.from_values_and_labels(values, labels, implementation=LAZY)
        ```
    """
    if enable is None:
        # Get current value
        return _use_eager_evaluation_for_lazy_arrays.get()
    else:
        # Return context manager
        @contextmanager
        def _context_manager() -> Generator[None, None, None]:
            token = _use_eager_evaluation_for_lazy_arrays.set(enable)
            try:
                yield
            finally:
                _use_eager_evaluation_for_lazy_arrays.reset(token)

        return _context_manager()


__all__ = [
    "auto_align",
    "use_eager_evaluation_for_lazy_arrays",
]

"""Global configuration settings for polder."""

from contextlib import _GeneratorContextManager, contextmanager
from contextvars import ContextVar
from typing import Generator, overload

# Global context variable to control auto-alignment behavior
_auto_align: ContextVar[bool] = ContextVar("auto_align", default=True)


@overload
def auto_align() -> bool:
    """Get the current auto_align setting."""
    ...


@overload
def auto_align(enable: bool) -> _GeneratorContextManager[None, None, None]:
    """Context manager to temporarily set the auto_align setting."""
    ...


def auto_align(enable: bool | None = None):
    """Get or set the auto_align setting.

    When called without arguments, returns the current auto_align value.
    When called with an argument, returns a context manager for temporary settings.

    Args:
        enable: Optional. Whether to enable or disable automatic alignment of arrays in binary operations.
                When False, alignment is only checked but not performed.
                If None (default), returns the current value instead.

    Returns:
        If enable is None: bool - the current auto_align setting
        If enable is bool: context manager for temporary changes

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


__all__ = [
    "auto_align",
]

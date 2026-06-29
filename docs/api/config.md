# Configuration

Global settings that change how operations behave. They are reached through the `config`
module, as `pld.config.<setting>`. Each setting is a function that returns the current value
when called without arguments, and returns a context manager when called with a boolean, so
you can scope a change to a block. See [Alignment](../guide/alignment.md) and
[Lazy arrays](../guide/lazy-arrays.md) for where these matter.

::: polder.config.auto_align

::: polder.config.use_eager_evaluation_for_lazy_arrays

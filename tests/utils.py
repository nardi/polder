from enum import Enum
from typing import Literal, overload

import jax
import jax.numpy as jnp
import narwhals as nw
import numpy as np
import polars as pl

import polder as pld
from polder.protocols.array import FrameLabeledArray


class ValueArrayType(Enum):
    NUMPY = np.ndarray
    JAX = jax.Array


@overload
def cast_value_array(
    values, value_array_type: Literal[ValueArrayType.NUMPY]
) -> np.ndarray: ...


@overload
def cast_value_array(
    values, value_array_type: Literal[ValueArrayType.JAX]
) -> jax.Array: ...


def cast_value_array(values, value_array_type: ValueArrayType):
    match value_array_type:
        case ValueArrayType.NUMPY:
            return np.asarray(values)
        case ValueArrayType.JAX:
            return jnp.asarray(values)


@overload
def generate_random_array(
    value_array_type: Literal[ValueArrayType.NUMPY] = ...,
    *,
    shape: tuple[int, ...] = ...,
    seed: int = ...,
) -> FrameLabeledArray[nw.DataFrame, np.ndarray]: ...


@overload
def generate_random_array(
    value_array_type: Literal[ValueArrayType.JAX],
    *,
    shape: tuple[int, ...] = ...,
    seed: int = ...,
) -> FrameLabeledArray[nw.DataFrame, jax.Array]: ...


def generate_random_array(
    value_array_type: ValueArrayType = ValueArrayType.NUMPY,
    *,
    shape: tuple[int, ...] = (2, 3),
    seed: int = 123,
):
    """Generate a random array with controllable shape and a fixed seed for reproducibility."""
    np.random.seed(seed)
    values = cast_value_array(np.random.randn(*shape), value_array_type)

    labels = []
    for dim_size in shape:
        # Create a dataframe with random labels for each dimension
        label_data = {
            f"dim{i}": np.arange(dim_size)
            for i in range(min(2, dim_size))  # At least one column, up to two
        }
        if not label_data:
            label_data = {"index": np.arange(dim_size)}
        labels.append(pl.DataFrame(label_data))

    return pld.from_values_and_labels(values, labels)


def shuffle_labels(array, axes=None, seed=123):
    """Shuffle the labels of an array along specified axes to test alignment."""
    np.random.seed(seed)

    if axes is None:
        axes = list(range(len(array.shape())))
    elif not isinstance(axes, (list, tuple)):
        axes = [axes]

    # Create index shuffles for each axis
    indices = []
    for i, size in enumerate(array.shape()):
        if i in axes:
            idx = np.random.permutation(size)
        else:
            idx = np.arange(size)
        indices.append(idx)

    # Apply the indexing
    return array[tuple(indices)]

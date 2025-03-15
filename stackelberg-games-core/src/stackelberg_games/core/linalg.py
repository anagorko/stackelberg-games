"""
This module implements linear algebra helper functions.
"""

from __future__ import annotations
from typing import Sequence

import numpy as np
import numpy.typing as npt


def to_vector(
    sequence: Sequence[float | np.float64] | npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Converts a sequence to a column matrix, i.e. to a vector."""

    # noinspection PyUnresolvedReferences
    if not isinstance(sequence, np.ndarray) or sequence.dtype != np.float64:
        vector = np.array(sequence, dtype=np.float64).reshape((len(sequence), 1))
    else:
        if sequence.ndim == 1:
            vector = sequence.reshape((sequence.shape[0], 11))
        else:
            vector = sequence

    assert vector.ndim == 2 and vector.shape[1] == 1

    return vector


def to_matrix(array: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Converts an array to a matrix."""

    # noinspection PyUnresolvedReferences
    if not isinstance(array, np.ndarray) or array.dtype != np.float64:
        return np.array(array, dtype=np.float64)

    return array


def standard_basis(m: int, i: int) -> npt.NDArray[np.float64]:
    """Returns i-th vector from the standard basis of R^m."""

    assert 0 <= i < m

    a = np.zeros(m, dtype=np.float64)
    a[i] = 1.0
    return a


def augment_vector(vector: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Returns an augmented vector."""

    assert vector.ndim == 2 and vector.shape[1] == 1

    return np.vstack((vector, np.array([[1.0]], dtype=np.float64)), dtype=np.float64)


def affine_transformation_matrix(
    linear: npt.ArrayLike | npt.NDArray[np.float64],
    translation: npt.ArrayLike | None = None,
) -> npt.NDArray[np.float64]:
    """Augments an affine transformation."""
    if not isinstance(linear, np.ndarray):
        linear = np.array(linear, dtype=np.float64)
    if linear.dtype != np.float64:
        linear = linear.astype(np.float64)

    if linear.ndim != 2:
        msg = f"Wrong number of dimension in the linear part ({linear.ndim} != 2)."
        raise ValueError(msg)

    m, n = linear.shape

    if translation is None:
        translation = np.zeros((m, 1), dtype=np.float64)
    else:
        if not isinstance(translation, np.ndarray):
            translation = np.array(translation, dtype=np.float64)
        if translation.dtype != np.float64:
            translation = translation.astype(np.float64)

        translation = translation.reshape((m, 1))

    return np.hstack(
        (
            np.vstack((linear, np.zeros((1, n), dtype=np.float64))),
            augment_vector(translation),
        )
    )

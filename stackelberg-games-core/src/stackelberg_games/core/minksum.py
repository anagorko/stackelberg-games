"""
This module computes vertices of Minkowski sums of polyhedra.
"""

from __future__ import annotations

import ast
import fractions
import itertools
import subprocess
import typing

import cdd  # type: ignore[import-not-found]
import numpy as np
import numpy.typing as npt
import scipy  # type: ignore[import-untyped]


def convex_hull(points: npt.NDArray[np.float64]) -> list[int]:
    """
    Convex hull computation.
    """
    return convex_hull_cdd(points)


def convex_hull_scipy(points: npt.NDArray[np.float64]) -> list[int]:
    """
    Convex hull computation with scipy library.
    """
    return typing.cast(
        list[int],
        np.array(scipy.spatial.ConvexHull(points.T).vertices, dtype=int).tolist(),
    )


def convex_hull_cdd(points: npt.NDArray[np.float64]) -> list[int]:
    """
    Convex hull computation with cdd library.
    """
    matrix = cdd.Matrix(np.hstack([np.ones((points.shape[1], 1)), points.T]))
    matrix.rep_type = cdd.RepType.GENERATOR
    _, redundancies = matrix.canonicalize()

    return [i for i in range(points.shape[1]) if i not in redundancies]


def effective_set(points: npt.NDArray[np.float64]) -> list[int]:
    """
    Effective set computation.
    """
    return effective_set_cdd(points)


def effective_set_cdd(points: npt.NDArray[np.float64]) -> list[int]:
    """
    Effective set computation with cdd library.
    """
    m, n = points.shape

    hull = np.hstack([np.ones((n, 1)), points.T])
    cone = np.hstack([np.zeros((m, 1)), -np.identity(m)])

    matrix = cdd.Matrix(np.array(np.vstack([hull, cone])))
    matrix.rep_type = cdd.RepType.GENERATOR
    _, redundancies = matrix.canonicalize()

    return [i for i in range(n) if i not in redundancies]


def effective_set_scipy(points: npt.NDArray[np.float64]) -> list[int]:
    hull = scipy.spatial.ConvexHull(points.T)
    vertices = set()
    faces = []
    for i, face in enumerate(hull.equations):
        if all(face[:-1] >= 0):
            faces.append(hull.points[hull.simplices[i]].T)
            vertices.update(set(hull.simplices[i]))
    return list(vertices)


def mink_sum_minksum(
    p1: npt.NDArray[np.float64], p2: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """
    Computes P_1 + P_2. P_1 and P_2 are convex hulls of columns of p1, p2.

    This implementation uses MinkSum executable:
        https://sites.google.com/site/christopheweibel/research/minksum
    """
    assert p1.shape[0] == p2.shape[0], (
        "Can't compute Minkowski sum of polyhedra "
        "with ambient spaces of different dimensions"
    )

    def format_matrix(
        v: list[float] | npt.NDArray[np.float64] | float | fractions.Fraction,
    ) -> str:
        if isinstance(v, list | np.ndarray):
            return f'[{",".join([format_matrix(n) for n in v])}]'

        if isinstance(v, float):
            v = fractions.Fraction(*v.as_integer_ratio())
        return str(v)

    input_data_str = (
        f"[{format_matrix(p1.transpose())}, " f"{format_matrix(p2.transpose())}]"
    )

    result = subprocess.run(
        ["minkSum", "-m", "-s"],
        input=input_data_str,
        encoding="utf8",
        capture_output=True,
        check=False,
    )

    parsed_result = eval(ast.unparse(ast.parse(result.stdout)))
    return np.array(parsed_result, dtype=np.float64).transpose()


def mink_sum_cdd(
    p1: npt.NDArray[np.float64], p2: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """
    Computes P_1 + P_2. P_1 and P_2 are convex hulls of columns of p1, p2.

    This implementation uses cdd.Matrix.canonicalize().
    """
    assert p1.shape[0] == p2.shape[0], (
        "Can't compute Minkowski sum of polyhedra "
        "with ambient spaces of different dimensions"
    )

    span = np.zeros((p1.shape[1] * p2.shape[1], 1 + p1.shape[0]))
    for i, (v1, v2) in enumerate(itertools.product(p1.transpose(), p2.transpose())):
        span[i][0] = 1
        span[i][1:] = v1 + v2

    hull = cdd.Matrix(span)
    hull.rep_type = cdd.RepType.GENERATOR
    hull.canonicalize()
    # noinspection PyTypeChecker
    return np.array(list(hull))[:, 1:].transpose()

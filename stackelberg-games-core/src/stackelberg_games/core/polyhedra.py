import fractions
import json
import subprocess
import typing
from collections.abc import Sequence

# noinspection PyPackageRequirements
import cdd  # type: ignore[import-not-found]
import numpy as np
import numpy.typing as npt

import math

from . import linalg


def embedding(points: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Returns augmented matrix [A|b] of isometry R^k -> span(points), where k = dim(span(points))."""

    basepoint = np.array(points[0], dtype=np.float64)
    span = np.array([point - basepoint for point in points[1:]], dtype=np.float64)

    k = np.linalg.matrix_rank(span)
    orthogonal_basis = np.linalg.qr(span.T).Q[:, 0:k]

    return np.hstack(
        (orthogonal_basis, np.reshape(basepoint, (basepoint.shape[0], 1))),
        dtype=np.float64,
    )


def subspace_isometry(
    points: npt.NDArray[np.float64],
) -> typing.Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]:
    """Returns an isometry span(points) -> R^k, where k = dim(span(points))."""

    basepoint = np.array(points[0], dtype=float)
    span = np.array([point - basepoint for point in points[1:]], dtype=float)

    k = np.linalg.matrix_rank(span)
    orthogonal_basis = np.linalg.qr(span.T).Q[:, 0:k]

    def isometry(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        v = x - basepoint
        return np.array(orthogonal_basis.T @ v, dtype=np.float64)

    return isometry


def random_point(
    rng: np.random.Generator, m: int, resolution: int = 47
) -> npt.NDArray[np.float64]:
    """Returns a discrete random point from the standard (m-1)-dimensional simplex."""

    x = rng.dirichlet([1] * m)
    x_rational: list[int | fractions.Fraction] = [
        fractions.Fraction(round(a * resolution), resolution) for a in x
    ]
    x_rational[-1] = 1 - sum(x_rational[:-1])

    return np.array(x_rational, dtype=np.float64)


def hyperplane(points: npt.NDArray[np.float64]) -> cdd.Polyhedron:
    """Returns hyperplane spanned by rows of 'points' matrix."""

    points = np.array(points)

    basepoint = points[-1]
    span = [points[i] - points[-1] for i in range(points.shape[0] - 1)]

    rows = np.vstack((basepoint, span))
    row_type = np.zeros((points.shape[0], 1))
    row_type[0] = 1

    generators = cdd.Matrix(
        np.hstack((row_type, rows)), number_type="float", linear=True
    )
    generators.rep_type = cdd.RepType.GENERATOR

    return cdd.Polyhedron(generators)


def ccw_sort(points: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Sorts points in 2d counter-clockwise.

    https://stackoverflow.com/questions/73683410/sort-vertices-of-a-convex-polygon-in-clockwise-or-counter-clockwise
    """

    assert points.shape[0] == 2

    lowest = min(points.T, key=lambda x: (x[1], x[0]))
    return np.array(
        sorted(points.T, key=lambda x: math.atan2(x[1] - lowest[1], -x[0] + lowest[0]))
    ).T


def m_simplex(m: int) -> cdd.Polyhedron:
    """Returns the standard (m-1)-dimensional simplex embedded in R^m."""

    generators = cdd.Matrix(
        np.hstack((np.ones((m, 1)), np.identity(m))), number_type="float"
    )
    generators.rep_type = cdd.RepType.GENERATOR

    return cdd.Polyhedron(generators)


def homog(p: cdd.Polyhedron) -> cdd.Polyhedron:
    """Returns homogenization of polyhedron `p`."""

    p_matrix = p.get_generators()
    p_array = np.array(p_matrix, dtype=np.float64)

    lin = p_array[tuple(p_matrix.lin_set), :]
    lin = np.hstack((lin, np.zeros((lin.shape[0], 1))))

    non_lin = p_array[tuple(set(range(p_array.shape[0])) - p_matrix.lin_set), 1:]
    non_lin = np.hstack(
        (np.ones((non_lin.shape[0], 1)), non_lin, np.ones((non_lin.shape[0], 1)))
    )
    non_lin[:, 0] = 0
    lin[:, 0] = 0

    p_homog = cdd.Matrix([linalg.standard_basis(p_array.shape[1] + 1, 0)], linear=False)
    p_homog.extend(non_lin, linear=False)
    if lin.size > 0:
        p_homog.extend(lin, linear=False)
    p_homog.rep_type = cdd.RepType.GENERATOR

    return cdd.Polyhedron(p_homog)


def vertices(p: cdd.Polyhedron) -> npt.NDArray[np.float64]:
    """Returns vertices of 'p' as rows of the returned array."""

    p_mat = np.array(p.get_generators(), dtype=np.float64)
    return p_mat[p_mat[:, 0] == 1, 1:]


def rays(p: cdd.Polyhedron) -> npt.NDArray[np.float64]:
    """Returns rays of 'p' as rows of the returned array."""

    p_mat = p.get_generators()
    p_array = np.array(p_mat, dtype=np.float64)

    return np.vstack(
        (p_array[p_array[:, 0] == 0, 1:], -p_array[tuple(p_mat.lin_set), 1:])
    )


def polar(p: cdd.Polyhedron) -> cdd.Polyhedron:
    """Returns the polar set for polyhedron 'p'."""
    p_vertices = vertices(p)
    p_rays = rays(p)

    q = cdd.Polyhedron(
        cdd.Matrix(
            np.vstack(
                (
                    np.hstack((np.ones((p_vertices.shape[0], 1)), -p_vertices)),
                    np.hstack((np.zeros((p_rays.shape[0], 1)), -p_rays)),
                )
            )
        )
    )
    q.rep_type = cdd.RepType.INEQUALITY
    return q


def intersection(p: cdd.Polyhedron, q: cdd.Polyhedron) -> cdd.Polyhedron:
    """Returns intersection of polyhedron 'p' with polyhedron 'q'."""

    hp = p.get_inequalities()
    hq = q.get_inequalities()

    lin_set = hq.lin_set
    ineq_set = set(range(hq.row_size)) - lin_set

    hq_np = np.array(hq)
    hq_lin = hq_np[list(lin_set), :]
    hq_ineq = hq_np[list(ineq_set), :]

    poly = hp.copy()
    if hq_ineq.size > 0:
        poly.extend(hq_ineq, linear=False)
    poly.extend(hq_lin, linear=True)
    poly.rep_type = cdd.RepType.INEQUALITY

    return cdd.Polyhedron(poly)


def image(embedding: npt.ArrayLike) -> cdd.Polyhedron:
    """Returns hyperplane spanned by rows of 'points' matrix."""

    embedding = np.array(embedding, dtype=np.float64)

    m, n = embedding.shape
    m -= 1
    n -= 1

    points = (
        linalg.affine_transformation_matrix(np.identity(m, dtype=np.float64))
        @ embedding
    )

    basepoint = points[-1]
    span = [points[i] - points[-1] for i in range(points.shape[0] - 1)]

    rows = np.vstack((basepoint, span))
    row_type = np.zeros((points.shape[0], 1))
    row_type[0] = 1

    generators = cdd.Matrix(
        np.hstack((row_type, rows)), number_type="float", linear=True
    )
    generators.rep_type = cdd.RepType.GENERATOR

    return cdd.Polyhedron(generators)


def preimage(ab: npt.NDArray[np.float64], p: cdd.Polyhedron) -> cdd.Polyhedron:
    """
    ab - augmented matrix [A | b] of embedding R^d -> R^m
    """
    h_matrix = p.get_inequalities()
    h_np_matrix = np.array(h_matrix)

    A = ab[:, :-1]
    b = np.reshape(ab[:, -1], (ab.shape[0], 1))

    C = h_np_matrix[:, 1:]
    d = np.reshape(h_np_matrix[:, 0], (C.shape[0], 1))

    preimage_matrix = cdd.Matrix(np.hstack((C @ b + d, C @ A)))
    preimage_matrix.lin_set = h_matrix.lin_set

    poly = cdd.Polyhedron(preimage_matrix)
    poly.rep_type = cdd.RepType.INEQUALITY
    return poly


def minksum(
    polytopes: tuple[npt.NDArray[np.float64], ...],
) -> tuple[tuple[int, ...], ...]:
    def format_matrix(
        v: tuple[npt.NDArray[np.float64], ...]
        | list[float]
        | npt.NDArray[np.float64]
        | float
        | fractions.Fraction,
    ) -> str:
        if isinstance(v, tuple | list | np.ndarray):
            return f'[{",".join([format_matrix(n) for n in v])}]'

        if isinstance(v, float):
            v = fractions.Fraction(*v.as_integer_ratio())
        return str(v)

    input_data_str = format_matrix(polytopes)

    # print(input_data_str)

    result = subprocess.run(
        ["minkSum", "-m"],
        input=input_data_str,
        encoding="utf8",
        capture_output=True,
        check=False,
    )

    parsed_result = [line.split(" ")[0] for line in result.stdout.split("\n")]
    return tuple(tuple(json.loads(line)) for line in parsed_result if line)


def cone(poly: cdd.Polyhedron) -> cdd.Polyhedron:
    """Returns cone of 'poly'."""

    v_repr = poly.get_generators()
    v_repr[:, 0] = 0

    assert v_repr.rep_type == cdd.RepType.GENERATOR

    return cdd.Polyhedron(v_repr)


def purify_cone(cone: cdd.Polyhedron) -> cdd.Polyhedron:
    """
    Remove vertex (0, 0, ...) from a conical set if needed.
    """

    cone_v_repr = cone.get_generators()
    cone_v_matrix = np.array(cone_v_repr)

    # If cone has a vertex, it has to be (0, 0, ..., 0)
    assert (cone_v_matrix[cone_v_matrix[:, 0] == 1, 1:] == 0).all()

    nonnegspan = []
    linspan = []
    for i in range(cone_v_matrix.shape[0]):
        if cone_v_matrix[i, 0] == 1:
            continue
        if i in cone_v_repr.lin_set:
            linspan.append(cone_v_matrix[i])
        else:
            nonnegspan.append(cone_v_matrix[i])

    if linspan:
        matrix = cdd.Matrix(linspan, linear=True)
        if nonnegspan:
            matrix.extend(nonnegspan, linear=False)
    else:
        matrix = cdd.Matrix(nonnegspan, linear=False)

    matrix.rep_type = cdd.RepType.GENERATOR

    return cdd.Polyhedron(matrix)


def convex_hull_cdd(points: npt.NDArray[np.float64]) -> list[int]:
    """
    Convex hull computation with cdd library.
    """
    matrix = cdd.Matrix(np.hstack([np.ones((points.shape[1], 1)), points.T]))
    matrix.rep_type = cdd.RepType.GENERATOR
    _, redundancies = matrix.canonicalize()

    return [i for i in range(points.shape[1]) if i not in redundancies]


def coordinates_of_extremal_points(
    polytopes: Sequence[npt.NDArray[np.float64]],
) -> list[tuple[int, ...]]:
    r"""Computes coordinates of extremal points of Minkowski sum

    \bigoplus_t \{ C^t e_i \colon 1 \leq i \leq n_t \}

    where n_t is the number of columns of C^t.
    """

    assert all(poly.ndim == 2 for poly in polytopes)
    assert all(poly.shape[0] == polytopes[0].shape[0] for poly in polytopes)

    # We remove redundant vertices from 'polytopes' first to work around some bugs in minkSum
    # cf. running minkSum on [[[1,3],[2,7],[3,6],[4,5],[5,2],[2,3],[3,5],[4,1]]]

    hulls_coordinates = [convex_hull_cdd(polytope) for polytope in polytopes]
    hulls_polytopes = [
        polytope[:, coordinates]
        for polytope, coordinates in zip(polytopes, hulls_coordinates, strict=False)
    ]

    def format_matrix(
        v: tuple[npt.NDArray[np.float64], ...]
        | list[np.float64 | float]
        | npt.NDArray[np.float64]
        | np.float64
        | float
        | fractions.Fraction,
    ) -> str:
        if isinstance(v, tuple | list | np.ndarray):
            return f'[{",".join([format_matrix(n) for n in v])}]'

        if isinstance(v, float):
            v = fractions.Fraction(*v.as_integer_ratio())
        return str(v)

    input_data_str = format_matrix(list(map(np.transpose, hulls_polytopes)))  # type: ignore[arg-type]
    result = subprocess.run(
        ["minkSum", "-m"],
        input=input_data_str,
        encoding="utf8",
        capture_output=True,
        check=False,
    )

    if result.stderr:
        msg = f"minkSum error: {result.stderr}"
        raise ValueError(msg)

    parsed_result = [line.split(" ")[0] for line in result.stdout.split("\n")]
    return [
        tuple(
            hulls_coordinates[k][coordinate]
            for k, coordinate in enumerate(json.loads(line))
        )
        for line in parsed_result
        if line
    ]


class MinkSum:
    r"""
    Represents polyhedra that are Minkowski sum of the following form

    \cone \{ x_1, \x_2, \ldots, x_m \} \oplus \bigoplus_t \conv \{ C^t_1, C^t_2, \ldots, C^t_{n_t} \}

    and provides access to vertices in decomposed form

    C^1_{i_1} + C^2_{i_2} + \cdots + C^t_{i_t}

    represented as tuples (i_1, i_2, \ldots, i_t).
    """

    def __init__(self, polytopes: Sequence[npt.NDArray[np.float64]]):
        self.polytopes = polytopes
        self.extremal_coordinates = coordinates_of_extremal_points(polytopes)

    def with_cone(self, cone: cdd.Polyhedron) -> Sequence[tuple[int, ...]]:
        """Computes coordinates of the Minkowski sum of the polytopes with a given cone."""

        cone = purify_cone(cone)
        cone_v_repr = cone.get_generators()

        effective_set = []
        for indices in self.extremal_coordinates:
            vertex = sum(
                poly[:, i] for i, poly in zip(indices, self.polytopes, strict=False)
            )
            effective_set.append(
                (1, *list(typing.cast(npt.NDArray[np.float64], vertex)))
            )

        offset = cone_v_repr.row_size
        cone_v_repr.extend(effective_set, linear=False)
        _, redundancies = cone_v_repr.canonicalize()
        return [
            self.extremal_coordinates[i]
            for i in range(len(effective_set))
            if i + offset not in redundancies
        ]

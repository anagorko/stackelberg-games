"""
This module implements functions that plot inducible actions polyhedra in 2d and 3d.
"""

from __future__ import annotations

import math

import cdd  # type: ignore[import-not-found]
import matplotlib as mpl
import matplotlib.axes
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.art3d as a3  # type: ignore[import-untyped]
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import numpy.typing as npt

from .. import minksum


def convex_hull_edges(points: npt.NDArray[np.float64]) -> list[tuple[int, int]]:
    m, n = points.shape

    hull = np.hstack([np.ones((n, 1)), points.T])
    matrix = cdd.Matrix(np.array(hull))
    polyhedron = cdd.Polyhedron(matrix)

    edges = []
    for i, adjacent in enumerate(polyhedron.get_input_adjacency()):
        for j in adjacent:
            if i > j:
                continue
            if matrix[i][0] == 1 and matrix[j][0] == 1:
                edges.append((i, j))

    return edges


def effective_set_edges(points: npt.NDArray[np.float64]) -> list[tuple[int, int]]:
    m, n = points.shape

    hull = np.hstack([np.ones((n, 1)), points.T])
    cone = np.hstack([np.zeros((m, 1)), -np.identity(m)])

    matrix = cdd.Matrix(np.array(np.vstack([hull, cone])))
    matrix.rep_type = cdd.RepType.GENERATOR

    polyhedron = cdd.Polyhedron(matrix)

    edges = []
    for i, adjacent in enumerate(polyhedron.get_input_adjacency()):
        for j in adjacent:
            if i > j:
                continue
            if matrix[i][0] == 1 and matrix[j][0] == 1:
                edges.append((i, j))

    return edges


def effective_set_rays(points: npt.NDArray[np.float64]) -> list[tuple[int, int]]:
    m, n = points.shape

    hull = np.hstack([np.ones((n, 1)), points.T])
    cone = np.hstack([np.zeros((m, 1)), -np.identity(m)])

    matrix = cdd.Matrix(np.array(np.vstack([hull, cone])))
    matrix.rep_type = cdd.RepType.GENERATOR

    polyhedron = cdd.Polyhedron(matrix)

    rays = []
    for i, adjacent in enumerate(polyhedron.get_input_adjacency()):
        for j in adjacent:
            if matrix[i][0] == 1 and matrix[j][0] == 0:
                dim = np.where(np.array(matrix[j]) == -1)[0][0] - 1
                rays.append((i, dim))

    return rays


def plot_effective_set(ax, points, color="blue"):
    effective_set = points[:, minksum.effective_set(points)]
    edges = effective_set_edges(effective_set)

    if effective_set.shape[0] == 2:
        ax.plot(*effective_set, ".", color=color, alpha=0.5, ms=1)
        for edge in edges:
            ax.add_line(
                plt.Line2D(*effective_set[:, edge], color=color, alpha=0.5, linewidth=1)
            )
    elif effective_set.shape[0] == 3:
        ax.plot(*effective_set, ".", color=color, alpha=0.5, ms=1)
        for edge in edges:
            ax.add_line(
                a3.Line3D(*effective_set[:, edge], color=color, alpha=0.5, linewidth=1)
            )
    else:
        msg = f"Plotting in dimension {effective_set.shape[0]} is not supported."
        raise ValueError(msg)


def plot_effective_rays(ax, points, color="blue"):
    effective_set = points[:, minksum.effective_set(points)]
    rays = effective_set_rays(effective_set)

    boundary = np.min(points, axis=1)

    for vertex, dim in rays:
        u = effective_set[:, vertex]
        v = np.array(u)
        v[dim] = boundary[dim] - 0.25

        if effective_set.shape[0] == 2:
            ax.add_line(
                plt.Line2D(
                    [u[0], v[0]],
                    [u[1], v[1]],
                    color=color,
                    linestyle="--",
                    linewidth=1,
                    alpha=0.5,
                )
            )
        elif effective_set.shape[0] == 3:
            ax.add_line(
                a3.Line3D(
                    [u[0], v[0]],
                    [u[1], v[1]],
                    [u[2], v[2]],
                    color=color,
                    linestyle="--",
                    linewidth=1,
                    alpha=0.5,
                )
            )


def plot_convex_hull(ax, points, color="blue", label=None, show_points=True):
    hull = points[:, minksum.convex_hull(points)]
    edges = convex_hull_edges(hull)

    if hull.shape[0] == 2:
        if show_points:
            ax.plot(hull[0], hull[1], ".", color=color, alpha=0.3, ms=0.7)
        ax.add_patch(
            plt.Polygon(hull.T, closed=True, alpha=0.1, color=color, label=label)
        )
        for edge in edges:
            ax.add_line(plt.Line2D(*hull[:, edge], color=color, alpha=0.2))
    elif hull.shape[0] == 3:
        if show_points:
            ax.plot(hull[0], hull[1], hull[2], ".", color=color, alpha=0.3, ms=0.7)
        for edge in edges:
            ax.add_line(
                a3.Line3D(*hull[:, edge], color=color, alpha=0.6, linewidth=0.75)
            )
    else:
        msg = f"Plotting in dimension {hull.shape[0]} is not supported."
        raise ValueError(msg)


def plot_inducible_actions(
    points, name: str, hull_color: str, effective_set_color: str
):
    if points.shape[0] == 2:
        fig = plt.figure(figsize=(4, 3), dpi=300)
        ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
        plot_inducible_actions_2d(points, ax, hull_color, effective_set_color)
        fig.savefig(name, bbox_inches="tight")
    elif points.shape[0] == 3:
        fig = plt.figure(figsize=(4, 3), dpi=300)
        ax = fig.add_axes((0.0, 0.0, 1.0, 1.0), projection="3d")
        plot_inducible_actions_3d(points, ax, hull_color, effective_set_color)
        fig.savefig(name, bbox_inches="tight")
    else:
        msg = f"Plotting in dimension {points.shape[0]} is not supported."
        raise ValueError(msg)


def plot_inducible_actions_2d(
    points,
    ax: mpl.axes.Axes,
    hull_color: str = "gray",
    effective_set_color: str = "blue",
):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.scatter(x=points[0, :], y=points[1, :], s=10)
    for i in range(points.shape[1]):
        ax.text(points[0][i] + 0.02, points[1][i] + 0.02, f"$C_{i+1}$")

    plot_convex_hull(ax, points, color=hull_color)
    plot_effective_set(ax, points, color=effective_set_color)
    plot_effective_rays(ax, points, color=effective_set_color)

    ax.set_aspect("equal")
    start, end = ax.get_xlim()
    start = math.ceil(start)
    ax.xaxis.set_ticks(np.arange(start, end, 1))
    print(np.arange(start, end, 1))


def plot_inducible_actions_3d(
    points, ax: Axes3D, hull_color: str = "gray", effective_set_color: str = "blue"
):
    ax.view_init(elev=10, azim=25, roll=-20)

    # ax.xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    # ax.yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    # ax.zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))

    plot_convex_hull(ax, points, color=hull_color)
    plot_effective_set(ax, points, color=effective_set_color)
    plot_effective_rays(ax, points, color=effective_set_color)

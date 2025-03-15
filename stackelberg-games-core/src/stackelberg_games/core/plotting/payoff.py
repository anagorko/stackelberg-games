"""
This module implements functions that plot generalized Bayesian Stackelberg games payoffs.
"""

from __future__ import annotations
from typing import Sequence

# noinspection PyPackageRequirements
import matplotlib.figure
import mpl_toolkits.mplot3d as a3  # type: ignore[import-untyped]
import numpy as np
import numpy.typing as npt

from ..games.generalized import GeneralizedBayesianGame
from ..polyhedra import ccw_sort, homog, polar, MinkSum


def inducible_responses(gm: GeneralizedBayesianGame) -> Sequence[tuple[int, ...]]:
    ms = MinkSum(gm.attacker_payoffs)
    return ms.with_cone(polar(homog(gm.defender_mixed_strategies)))


def plot_payoff(
    ax: a3.Axes3D,
    gm: GeneralizedBayesianGame,
    payoffs: tuple[npt.NDArray[np.float64], ...],
    domain_height: float | None = None,
):
    domain_polys = []
    payoff_polys = []

    for i, response in enumerate(inducible_responses(gm)):
        region = gm.response_inducing_region(response)

        pts = np.array(region.get_generators())[:, 1:].T
        pts = np.hstack((ccw_sort(pts).T, np.zeros((pts.shape[1], 1))))

        payoff = sum(
            gm.attacker_distribution[t] * payoffs[t][:, response[t]]
            for t in range(gm.T)
        )

        domain_polys.append(pts)

        pts = np.copy(pts)
        pts[:, -1] = 1
        pts[:, -1] = pts @ payoff

        payoff_polys.append(pts)

    max_x = max(0, *(max(poly[:, 0]) for poly in payoff_polys))
    min_x = min(0, *(min(poly[:, 0]) for poly in payoff_polys))

    max_y = max(0, *(max(poly[:, 1]) for poly in payoff_polys))
    min_y = min(0, *(min(poly[:, 1]) for poly in payoff_polys))

    max_z = max(0, *(max(poly[:, 2]) for poly in payoff_polys))
    min_z = min(0, *(min(poly[:, 2]) for poly in payoff_polys))

    for domain_poly in domain_polys:
        if domain_height is None:
            domain_poly[:, -1] = min_z
        else:
            domain_poly[:, -1] = domain_height

    domain_plot = a3.art3d.Poly3DCollection(
        domain_polys, alpha=0.2, edgecolors=(0, 0, 0, 0.3)
    )
    ax.add_collection3d(domain_plot)

    payoff_plot = a3.art3d.Poly3DCollection(
        payoff_polys, facecolors="blue", edgecolors="k", shade=True, alpha=0.8
    )
    ax.add_collection3d(payoff_plot)

    ax.axes.set_xlim3d(left=min_x, right=max_x)
    ax.axes.set_ylim3d(bottom=min_y, top=max_y)
    ax.axes.set_zlim3d(bottom=min_z, top=max_z)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.view_init(elev=45, azim=60, roll=0)


def sync_axes(fig: matplotlib.figure.Figure, axes: list[a3.Axes3D]):
    def on_move(event):
        redraw = False
        for ax in axes:
            if event.inaxes == ax:
                for ax2 in axes:
                    if ax != ax2:
                        ax2.view_init(elev=ax.elev, azim=ax.azim)
                redraw = True
        if redraw:
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", on_move)


def interactive_plot(
    gm: GeneralizedBayesianGame, fig: matplotlib.figure.Figure
) -> None:
    ax_def = fig.add_subplot(1, 2, 1, projection="3d")
    ax_att = fig.add_subplot(1, 2, 2, projection="3d")

    plot_payoff(ax_def, gm, gm.defender_payoffs)
    ax_def.set_title("Defender Payoff")

    plot_payoff(ax_att, gm, gm.attacker_payoffs)
    ax_att.set_title("Attacker Payoff")

    sync_axes(fig, [ax_att, ax_def])
    fig.suptitle(f"Correlation coefficient {gm.correlation_coefficient():.4f}")

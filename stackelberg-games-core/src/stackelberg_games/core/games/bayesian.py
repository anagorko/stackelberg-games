"""
This module implements a data class for Bayesian games in a normal form.
"""

from __future__ import annotations
from itertools import product
from typing import Sequence


import numpy as np
import numpy.typing as npt
import pandas as pd
import pydantic

from .game import Game
from .normal_form import NormalFormGame


class BayesianGame(Game):
    """
    A Bayesian game in a normal form.
    """

    def __init__(
        self,
        defender_payoffs: Sequence[npt.ArrayLike],
        attacker_payoffs: Sequence[npt.ArrayLike],
        attacker_distribution: Sequence[float | np.float64],
        defender_moves: Sequence[str | int] | None = None,
        attackers_moves: Sequence[Sequence[str | int]] | None = None,
        defender_name: str | int | None = None,
        attacker_names: Sequence[str | int] | None = None,
    ) -> None:
        self.defender_payoffs: tuple[npt.NDArray[np.float64], ...] = tuple(
            np.array(defender_payoff, dtype=np.float64)
            if not isinstance(defender_payoff, np.ndarray)
            else defender_payoff
            for defender_payoff in defender_payoffs
        )
        self.attacker_payoffs: tuple[npt.NDArray[np.float64], ...] = tuple(
            np.array(attacker_payoff, dtype=np.float64)
            if not isinstance(attacker_payoff, np.ndarray)
            else attacker_payoff
            for attacker_payoff in attacker_payoffs
        )

        assert (
            len(self.defender_payoffs)
            == len(self.attacker_payoffs)
            == len(attacker_distribution)
        )
        assert all(
            attacker_payoff.shape == defender_payoff.shape
            for attacker_payoff, defender_payoff in zip(
                self.attacker_payoffs, self.defender_payoffs
            )
        )

        assert all(
            len(attacker_payoff.shape) == 2 for attacker_payoff in self.attacker_payoffs
        )
        assert all(
            attacker_payoff.shape == self.attacker_payoffs[0].shape
            for attacker_payoff in self.attacker_payoffs
        )

        self.attacker_distribution: npt.NDArray[np.float64] = np.array(
            attacker_distribution, dtype=np.float64
        )

        if defender_moves is None:
            self.defender_moves: tuple[int | str, ...] = tuple(
                range(self.defender_payoffs[0].shape[0])
            )
        else:
            self.defender_moves = tuple(defender_moves)

        if attackers_moves is None:
            self.attackers_moves: tuple[tuple[int | str, ...], ...] = tuple(
                tuple(range(attacker_payoff.shape[1]))
                for attacker_payoff in self.attacker_payoffs
            )
        else:
            self.attackers_moves = tuple(
                tuple(att_moves) for att_moves in attackers_moves
            )

        if defender_name is None:
            self.defender_name: str | int = "Alice"
        else:
            self.defender_name = defender_name

        if attacker_names is None:
            self.attacker_names: tuple[str | int, ...] = tuple(
                f"Bob{i}" for i in range(len(self.attacker_payoffs))
            )
        else:
            self.attacker_names = tuple(attacker_names)

    class BayesianGameModel(pydantic.BaseModel):
        model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

        defender_payoffs: tuple[list[list[float]], ...]
        attacker_payoffs: tuple[list[list[float]], ...]
        attacker_distribution: list[float]
        defender_moves: tuple[str | int, ...]
        attackers_moves: tuple[tuple[str | int, ...], ...]
        defender_name: str | int
        attacker_names: tuple[str | int, ...]

    def get_metadata(self) -> BayesianGame.BayesianGameModel:
        return BayesianGame.BayesianGameModel(
            defender_payoffs=tuple(
                defender_payoff.tolist() for defender_payoff in self.defender_payoffs
            ),
            attacker_payoffs=tuple(
                attacker_payoff.tolist() for attacker_payoff in self.attacker_payoffs
            ),
            attacker_distribution=self.attacker_distribution.tolist(),
            defender_moves=self.defender_moves,
            attackers_moves=self.attackers_moves,
            defender_name=self.defender_name,
            attacker_names=self.attacker_names,
        )

    @classmethod
    def from_metadata(cls, model: pydantic.BaseModel) -> BayesianGame:
        assert isinstance(model, BayesianGame.BayesianGameModel)

        return BayesianGame(
            tuple(
                np.array(defender_payoff, dtype=np.float64)
                for defender_payoff in model.defender_payoffs
            ),
            tuple(
                np.array(attacker_payoff, dtype=np.float64)
                for attacker_payoff in model.attacker_payoffs
            ),
            model.attacker_distribution,
            model.defender_moves,
            model.attackers_moves,
            model.defender_name,
            model.attacker_names,
        )

    @classmethod
    def from_dicts(
        cls,
        defender_moves: set[str | int],
        attacker_types: set[str | int],
        attackers_moves: dict[str | int, set[str | int]],
        defender_payoffs: dict[str | int, dict[str | int, dict[str | int, float]]],
        attacker_payoffs: dict[str | int, dict[str | int, dict[str | int, float]]],
        attacker_distribution: dict[str | int, float],
    ):
        defender_moves_enum = dict(enumerate(sorted(defender_moves)))
        attacker_types_enum = dict(enumerate(sorted(attacker_types)))
        attackers_moves_enum = {
            t: dict(enumerate(sorted(attackers_moves[attacker_types_enum[t]])))
            for t in attacker_types_enum
        }
        defender_payoffs_array = [
            np.array(
                [
                    [
                        defender_payoffs[attacker_types_enum[t]][
                            defender_moves_enum[i]
                        ][attackers_moves_enum[t][j]]
                        for j in attackers_moves_enum[t]
                    ]
                    for i in defender_moves_enum
                ]
            )
            for t in attacker_types_enum
        ]
        attacker_payoffs_array = [
            np.array(
                [
                    [
                        attacker_payoffs[attacker_types_enum[t]][
                            defender_moves_enum[i]
                        ][attackers_moves_enum[t][j]]
                        for j in attackers_moves_enum[t]
                    ]
                    for i in defender_moves_enum
                ]
            )
            for t in attacker_types_enum
        ]
        attacker_distribution_list = [
            attacker_distribution[attacker_types_enum[t]] for t in attacker_types_enum
        ]
        attacker_names = [str(name) for name in attacker_types_enum.values()]

        return BayesianGame(
            defender_payoffs_array,
            attacker_payoffs_array,
            attacker_distribution_list,
            list(defender_moves_enum.values()),
            [list(attackers_moves_enum[t].values()) for t in attacker_types_enum],
            "Alice",
            attacker_names,
        )

    class VerboseModel(pydantic.BaseModel):
        X: set[str | int]
        L: set[str | int]
        Q: dict[str | int, set[str | int]]
        R: dict[str | int, dict[str | int, dict[str | int, float]]]
        C: dict[str | int, dict[str | int, dict[str | int, float]]]
        p: dict[str | int, float]
        T: int
        m: int
        n: dict[str | int, int]

        def expected_reward(self, x: dict) -> float:
            payoff = 0
            for ll in self.L:
                gv = {}
                for q in self.Q[ll]:
                    gv[q] = sum(self.C[ll][i][q] * x[i] for i in self.X)
                payoffs = {}
                max_gv = max(gv.values())
                for q in self.Q[ll]:
                    if abs(gv[q] - max_gv) < 0.0000001:
                        payoffs[q] = sum(self.R[ll][i][q] * x[i] for i in self.X)
                payoff += self.p[ll] * max(payoffs.values())
            return payoff

    def to_dicts(self) -> VerboseModel:
        T = len(self.attacker_distribution)
        m = len(self.defender_moves)
        n: dict[str | int, int] = {t: len(self.attackers_moves[t]) for t in range(T)}

        defender_moves = set(self.defender_moves)
        attacker_types: set[str | int] = set(self.attacker_names)
        attackers_moves = {
            self.attacker_names[t]: set(self.attackers_moves[t]) for t in range(T)
        }
        defender_payoffs = {
            self.attacker_names[t]: {
                self.defender_moves[i]: {
                    self.attackers_moves[t][j]: float(self.defender_payoffs[t][i, j])
                    for j in range(n[t])
                }
                for i in range(m)
            }
            for t in range(T)
        }
        attacker_payoffs = {
            self.attacker_names[t]: {
                self.defender_moves[i]: {
                    self.attackers_moves[t][j]: float(self.attacker_payoffs[t][i, j])
                    for j in range(n[t])
                }
                for i in range(m)
            }
            for t in range(T)
        }
        attacker_distribution = {
            self.attacker_names[t]: self.attacker_distribution[t] for t in range(T)
        }
        return BayesianGame.VerboseModel(
            X=defender_moves,
            L=attacker_types,
            Q=attackers_moves,
            R=defender_payoffs,
            C=attacker_payoffs,
            p=attacker_distribution,
            T=T,
            m=m,
            n=n,
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BayesianGame):
            return False

        return (
            all(
                np.array_equal(self_defender_payoff, other_defender_payoff)
                for self_defender_payoff, other_defender_payoff in zip(
                    self.defender_payoffs, other.defender_payoffs
                )
            )
            and all(
                np.array_equal(self_attacker_payoff, other_attacker_payoff)
                for self_attacker_payoff, other_attacker_payoff in zip(
                    self.attacker_payoffs, other.attacker_payoffs
                )
            )
            and self.defender_moves == other.defender_moves
            and self.attackers_moves == other.attackers_moves
            and self.defender_name == other.defender_name
            and self.attacker_names == other.attacker_names
            and np.array_equal(self.attacker_distribution, other.attacker_distribution)
        )

    def correlation_coefficient(self) -> np.float64:
        """Computes a correlation coefficient between defender and attacker payoff matrices."""

        return sum(
            self.attacker_distribution[t]
            * (
                np.dot(
                    self.defender_payoffs[t].flatten(),
                    self.attacker_payoffs[t].flatten(),
                )
            )
            / np.linalg.norm(self.defender_payoffs[t].flatten())
            / np.linalg.norm(self.attacker_payoffs[t].flatten())
            for t in range(self.T)
        )

    def to_normal_form(self) -> NormalFormGame:
        """Converts a Bayesian game to a normal form using Harsanyi transformation."""

        J = tuple(
            tuple(zip(range(self.T), s))
            for s in product(*[range(self.n[t]) for t in range(self.T)])
        )
        """Follower moves after transformation are mappings L -> Q[l]"""

        def combined_payoff(payoffs) -> list[list[float]]:
            """
            :param payoffs: a Bayesian payoff matrix, i.e. mapping "follower type" -> "payoff matrix".
            :return: a mapping I x J -> "payoff"
            """
            return [
                [
                    sum(self.attacker_distribution[t] * payoffs[t][i, j] for t, j in s)
                    for s in J
                ]
                for i in range(self.m)
            ]

        defender_payoff = combined_payoff(self.defender_payoffs)
        attacker_payoff = combined_payoff(self.attacker_payoffs)

        attacker_moves = tuple(str(mv) for mv in J)

        return NormalFormGame(
            defender_payoff,
            attacker_payoff,
            self.defender_moves,
            attacker_moves,
            self.defender_name,
        )

    @classmethod
    def from_normal_form(cls, game: NormalFormGame):
        attacker_distribution_list = [1.0]
        attacker_names = ["Bob"]

        return BayesianGame(
            [game.defender_payoff],
            [game.attacker_payoff],
            attacker_distribution_list,
            game.defender_moves,
            [game.attacker_moves],
            "Alice",
            attacker_names,
        )

    def _repr_html_(self) -> str:
        """
        Html rendering of the game.
        """

        game = self.to_dicts()

        def latex(t):
            if isinstance(t, tuple):
                return ", ".join(latex(s) for s in t)
            elif isinstance(t, list):
                return ", ".join(latex(s) for s in t)
            elif t == r"\emptyset":
                return r"$\emptyset$"
            else:
                return f"{t}"

        html = [
            pd.DataFrame.from_dict(game.R[t], orient="index")
            .sort_index()
            .style.format(precision=2)
            .format_index(lambda kv: f"{latex(kv)}", axis=1)
            .set_properties(
                subset=None, **{"white-space": "nowrap", "word-break": "break-all"}
            )
            .set_caption(
                f"Leader payoff against attacker type {t} ({game.p[t]:.2f}) (R matrix)"
            )
            .set_table_styles(
                [
                    {
                        "selector": "caption",
                        "props": [("font-size", "16px"), ("font-weight", "bold")],
                    }
                ]
            )
            .to_html(sparse_columns=False, sparse_index=False, notebook=True)
            for t in game.L
        ] + [
            pd.DataFrame.from_dict(game.C[t], orient="index")
            .sort_index()
            .style.format(precision=2)
            .format_index(lambda kv: f"{latex(kv)}", axis=1)
            .set_properties(
                subset=None, **{"white-space": "nowrap", "word-break": "break-all"}
            )
            .set_caption(f"Follower {t} payoff (C matrix)")
            .set_table_styles(
                [
                    {
                        "selector": "caption",
                        "props": [("font-size", "16px"), ("font-weight", "bold")],
                    }
                ]
            )
            .to_html(sparse_columns=False, sparse_index=False, notebook=True)
            for t in game.L
        ]
        return "<br>".join(html)

    @property
    def m(self) -> int:
        return self.defender_payoffs[0].shape[0]

    @property
    def T(self) -> int:
        return len(self.attacker_distribution)

    @property
    def n(self) -> tuple[int, ...]:
        return tuple(self.defender_payoffs[i].shape[1] for i in range(self.T))

    def attacker_best_response(self, x: npt.NDArray[np.float64]) -> tuple[int, ...]:
        response = []
        for t in range(self.T):
            gv: dict[int, np.float64] = {}
            for j in range(self.n[t]):
                gv[j] = x.T @ self.attacker_payoffs[t][:, j]  # type: ignore[assignment]
            max_gv = max(gv.values())  # type: ignore[type-var]
            r = -1
            def_gv = float("-inf")
            for j in range(self.n[t]):
                if abs(gv[j] - max_gv) < 10**-7:
                    df = x.T @ self.defender_payoffs[t][:, j]
                    if df > def_gv:
                        def_gv = df  # type: ignore[assignment]
                        r = j
            response.append(r)
        return tuple(response)

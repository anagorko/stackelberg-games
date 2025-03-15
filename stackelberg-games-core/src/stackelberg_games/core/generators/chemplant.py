"""
Chemical Plant defending games, based on

Applying a Bayesian Stackelberg Game for
Securing a Chemical Plant

Laobing Zhanga, Genserik Reniers
"""

import itertools
from enum import Enum
from fractions import Fraction
from functools import reduce
from typing import cast

from ..games.bayesian import BayesianGame
from .generator import GameGenerator, GameGeneratorMetadata


class ChemicalPlantAttacker(Enum):
    TERRORIST = 1
    ACTIVIST = 2
    CRIMINAL = 3


# Table A.1
class Targets(Enum):
    T1 = 1
    T2 = 2
    T3 = 3
    T4 = 4
    T5 = 5
    T6 = 6


# Table A.1
class EntryPoint(Enum):
    MAINGATE = 1
    DOCK = 2
    GATE = 3
    ENCLOSE1 = 4
    ENCLOSE2 = 5
    ANYWHERE = 6


class ChemicalPlantGameGenerator(GameGenerator):
    tag: str = "chemplant"

    class AttackerStrategy:
        def __init__(self, target, perimeters, effort, att_type):
            self.target = target
            self.perimeters = perimeters
            self.effort = effort
            self.zone_level = len(perimeters)
            self.att_type = att_type

    class DefenderStrategy:
        def __init__(self, zone_alert_levels, perimeter_alert_levels):
            self.zone_alert_levels = zone_alert_levels
            self.perimeter_alert_levels = perimeter_alert_levels

    @staticmethod
    def _init_effort_cost():
        ec = {
            ChemicalPlantAttacker.TERRORIST: {0: 0, 1: 10, 2: 20, 3: 30},
            ChemicalPlantAttacker.ACTIVIST: {0: 0, 1: 6, 2: 12, 3: 24},
        }
        return ec

    @staticmethod
    def _init_cost_reward():
        L = {
            ChemicalPlantAttacker.TERRORIST: {
                Targets.T1: (1000, 1000),
                Targets.T2: (100, 100),
                Targets.T3: (300, 300),
                Targets.T4: (800, 800),
                Targets.T5: (2000, 3000),
                Targets.T6: (10000, 8000),
            },
            ChemicalPlantAttacker.ACTIVIST: {
                Targets.T1: (0, 0),
                Targets.T2: (100, 100),
                Targets.T3: (300, 300),
                Targets.T4: (800, 800),
                Targets.T5: (0, 0),
                Targets.T6: (1000, 1000),
            },
        }
        return L

    @staticmethod
    def _init_time_table():
        return {
            0: {
                EntryPoint.ANYWHERE: {
                    Targets.T1: 5,
                    Targets.T2: 5,
                    EntryPoint.MAINGATE: 5,
                    EntryPoint.DOCK: 5,
                    EntryPoint.ENCLOSE1: 5,
                }
            },
            1: {
                EntryPoint.MAINGATE: {
                    Targets.T3: 10,
                    Targets.T4: 10,
                    Targets.T5: 15,
                    EntryPoint.GATE: 10,
                    EntryPoint.ENCLOSE2: 5,
                },
                EntryPoint.DOCK: {
                    Targets.T3: 6,
                    Targets.T4: 7,
                    Targets.T5: 3,
                    EntryPoint.GATE: 10,
                    EntryPoint.ENCLOSE2: 12,
                },
                EntryPoint.ENCLOSE1: {
                    Targets.T3: 5,
                    Targets.T4: 5,
                    Targets.T5: 5,
                    EntryPoint.GATE: 5,
                    EntryPoint.ENCLOSE2: 5,
                },
            },
            2: {EntryPoint.GATE: {Targets.T6: 3}, EntryPoint.ENCLOSE2: {Targets.T6: 3}},
        }

    def _init_zone_defence_cost(self):
        return [[40, 60, 100], [20, 30, 50], [20, 30, 50]]

    def _init_entry_defence_cost(self):
        return [
            [20, 30, 50],  # EntryPoint.MAINGATE
            [20, 25, 40],  # EntryPoint.DOCK
            [20, 30, 50],  # EntryPoint.ENCLOSE1
            [20, 25, 40],  # EntryPoint.GATE
            [10, 20, 40],  # EntryPoint.ENCLOSE2
        ]

    def get_metadata(self) -> GameGeneratorMetadata:
        """Serializes the Generator instance to a pydantic.BaseModel."""
        return GameGeneratorMetadata(
            cls=self.__class__,
            args=(),
            kwargs={
                "attacker_types": self.attacker_types,
                "alpha_z": self.alpha_z,
                "tau": self.tau,
                "beta_p": self.beta_p,
            },
        )

    def __init__(
        self,
        attacker_types: list[tuple[ChemicalPlantAttacker, Fraction]] | None = None,
        alpha_z: list[list[int]] | None = None,
        tau: int = 5,
        beta_p: dict[int, dict[EntryPoint, int]] | None = None,
    ):
        if attacker_types is None:
            self.attacker_types: list[tuple[ChemicalPlantAttacker, Fraction]] = [
                (ChemicalPlantAttacker.ACTIVIST, Fraction(4, 7)),
                (ChemicalPlantAttacker.TERRORIST, Fraction(3, 7)),
            ]
        if alpha_z is None:
            self.alpha_z = [[1], [3], [6]]
        else:
            self.alpha_z = alpha_z
        if beta_p is None:
            self.beta_p: dict[int, dict[EntryPoint, int]] = {
                1: {EntryPoint.MAINGATE: 1, EntryPoint.DOCK: 3, EntryPoint.ENCLOSE1: 2},
                2: {EntryPoint.GATE: 2, EntryPoint.ENCLOSE2: 3},
            }
        else:
            self.beta_p = beta_p

        self.destr_prob = {
            Targets.T1: 0.1,
            Targets.T2: 0.9,
            Targets.T3: 0.7,
            Targets.T4: 0.6,
            Targets.T5: 0.9,
            Targets.T6: 0.99,
        }
        self.tau = tau
        self.time_table = self._init_time_table()
        self.cost_reward = self._init_cost_reward()
        self.effort_cost = self._init_effort_cost()
        self.zone_defence_cost = self._init_zone_defence_cost()
        self.entry_defence_cost = self._init_entry_defence_cost()
        self.zone_to_target_map = {
            0: {Targets.T1, Targets.T2},
            1: {Targets.T3, Targets.T4, Targets.T5},
            2: {Targets.T6},
        }
        self.zone_to_entrance_map = {
            0: {EntryPoint.ANYWHERE},
            1: {EntryPoint.MAINGATE, EntryPoint.DOCK, EntryPoint.ENCLOSE1},
            2: {EntryPoint.GATE, EntryPoint.ENCLOSE2},
        }

    # so-called contest success function
    def _p_zone(self, level, sublevel, effort, alert_level, time_stayed):
        if effort == 0:
            return Fraction(0, 1)
        denominator = Fraction(
            effort
            + self.alpha_z[level][sublevel] * alert_level * time_stayed / self.tau
        )
        return Fraction(effort * denominator.denominator, denominator.numerator)

    def _p_perimeter(self, level, entrance, effort, alert_level):
        if effort == 0:
            return Fraction(0, 1)
        denominator = effort + self.beta_p[level][entrance] * alert_level
        return Fraction(effort * denominator.denominator, denominator.numerator)

    def _calculate_p(
        self, attacker_strategy: AttackerStrategy, defender_strategy: DefenderStrategy
    ):
        p = list()
        for lvl in range(attacker_strategy.zone_level):
            entrance_from = attacker_strategy.perimeters[lvl]
            target_or_entrance = (
                attacker_strategy.target
                if lvl == attacker_strategy.zone_level - 1
                else attacker_strategy.perimeters[lvl + 1]
            )
            p_zone = self._p_zone(
                lvl,
                0,
                attacker_strategy.effort,
                defender_strategy.zone_alert_levels[lvl],
                self.time_table[lvl][entrance_from][target_or_entrance],
            )
            p.append(p_zone)

        for lvl in range(1, attacker_strategy.zone_level):
            p_perim = self._p_perimeter(
                lvl,
                attacker_strategy.perimeters[lvl],
                attacker_strategy.effort,
                defender_strategy.perimeter_alert_levels[lvl],
            )
            p.append(p_perim)
        res_p = reduce((lambda x, y: x * y), p)
        return res_p

    def _attacker_utility(
        self, attacker_strategy: AttackerStrategy, defender_strategy: DefenderStrategy
    ):
        res_p = self._calculate_p(attacker_strategy, defender_strategy)

        return (
            res_p
            * self.destr_prob[attacker_strategy.target]
            * self.cost_reward[attacker_strategy.att_type][attacker_strategy.target][1]
            - self.effort_cost[attacker_strategy.att_type][attacker_strategy.effort]
        )

    def _defence_cost(self, defender_strategy: DefenderStrategy):
        defence_sum = 0.0
        for lvl in defender_strategy.perimeter_alert_levels:
            defence_sum += self.entry_defence_cost[lvl][
                defender_strategy.perimeter_alert_levels[lvl]
            ]
        for lvl in range(len(defender_strategy.zone_alert_levels)):
            defence_sum += self.zone_defence_cost[lvl][
                defender_strategy.zone_alert_levels[lvl]
            ]
        return defence_sum

    def _defender_utility(
        self, attacker_strategy: AttackerStrategy, defender_strategy: DefenderStrategy
    ):
        res_p = self._calculate_p(attacker_strategy, defender_strategy)

        return -(
            res_p
            * self.destr_prob[attacker_strategy.target]
            * self.cost_reward[attacker_strategy.att_type][attacker_strategy.target][0]
            + self._defence_cost(defender_strategy)
        )

    def get_instance(self, random_seed: int) -> BayesianGame:
        # defender moves
        alert_levels = 3
        zones = 3
        passings = 5
        zone_levels = [
            list(pair) for pair in itertools.product(range(alert_levels), repeat=zones)
        ]
        passings_levels = [
            list(pair)
            for pair in itertools.product(range(alert_levels), repeat=passings)
        ]
        defender_moves = [
            self.DefenderStrategy(zl, pl)
            for zl in zone_levels
            for pl in passings_levels
        ]
        X: set[str | int] = set(range(pow(alert_levels, zones + passings)))

        # attacker types
        L: set[str | int] = set(range(len(self.attacker_types)))
        # probability distribution on attacker types
        p: dict[str | int, float] = {
            t: float(self.attacker_types[t][1]) for t in cast(set[int], L)
        }
        # attacker moves
        # 3 effort levels
        attacker_targets = {
            ChemicalPlantAttacker.TERRORIST: {
                Targets.T1,
                Targets.T2,
                Targets.T3,
                Targets.T4,
                Targets.T5,
                Targets.T6,
            },
            ChemicalPlantAttacker.ACTIVIST: {
                Targets.T1,
                Targets.T2,
                Targets.T3,
                Targets.T4,
                Targets.T5,
                Targets.T6,
            },  # they need to have the same amount of moves because of logic in solvers
        }
        activist_moves = list()
        for z in range(3):
            zone_targets = self.zone_to_target_map[z]
            targets0 = attacker_targets[ChemicalPlantAttacker.ACTIVIST]
            targets = [t for t in targets0 if t in zone_targets]
            entrances = list()
            for i in range(z + 1):
                entrances.append(self.zone_to_entrance_map[i])
            paths = list(itertools.product(*entrances))
            activist_moves += [
                self.AttackerStrategy(t, p, e, ChemicalPlantAttacker.ACTIVIST)
                for t in targets
                for p in paths
                for e in range(4)
            ]

        terrorist_moves = list()
        for z in range(3):
            zone_targets = self.zone_to_target_map[z]
            targets0 = attacker_targets[ChemicalPlantAttacker.TERRORIST]
            targets = [t for t in targets0 if t in zone_targets]
            entrances = list()
            for i in range(z + 1):
                entrances.append(self.zone_to_entrance_map[i])
            paths = list(itertools.product(*entrances))
            terrorist_moves += [
                self.AttackerStrategy(t, p, e, ChemicalPlantAttacker.TERRORIST)
                for t in targets
                for p in paths
                for e in range(4)
            ]
        attacker_moves = {
            ChemicalPlantAttacker.ACTIVIST: activist_moves,
            ChemicalPlantAttacker.TERRORIST: terrorist_moves,
        }

        Q: dict[str | int, set[str | int]] = {
            t: set(range(len(attacker_moves[self.attacker_types[t][0]])))
            for t in cast(set[int], L)
        }

        def defender_payoff(t, i, j):
            attacker_type = self.attacker_types[t][0]
            defender_strategy = defender_moves[i]
            attacker_strategy = attacker_moves[attacker_type][j]
            return self._defender_utility(attacker_strategy, defender_strategy)

        def attacker_payoff(t, i, j):
            attacker_type = self.attacker_types[t][0]
            defender_strategy = defender_moves[i]
            attacker_strategy = attacker_moves[attacker_type][j]
            return self._attacker_utility(attacker_strategy, defender_strategy)

        R: dict[str | int, dict[str | int, dict[str | int, float]]] = {
            t: {i: {j: defender_payoff(t, i, j) for j in Q[t]} for i in X} for t in L
        }
        C: dict[str | int, dict[str | int, dict[str | int, float]]] = {
            t: {i: {j: attacker_payoff(t, i, j) for j in Q[t]} for i in X} for t in L
        }

        description = "Chemical Plant defence game"

        if random_seed is not None:
            description += f" (random seed {random_seed})."

        return BayesianGame.from_dicts(X, L, Q, R, C, p)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ChemicalPlantGameGenerator):
            return False

        return (
            self.attacker_types == other.attacker_types
            and self.alpha_z == other.alpha_z
            and self.tau == other.tau
            and self.beta_p == other.beta_p
        )

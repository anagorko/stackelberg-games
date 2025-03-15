"""
This module implements DOBSS (Decomposed Optimal Bayesian Stackelberg Solver) from

Paruchuri, Praveen, et al.

  "Playing games for security: An efficient exact algorithm for solving Bayesian Stackelberg games."

Proceedings of the 7th international joint conference on Autonomous agents and multiagent systems-Volume 2. 2008.
"""

from __future__ import annotations

import pydantic
import pyscipopt  # type: ignore[import-untyped]
import pyscipopt.scip  # type: ignore[import-untyped]
from tqdm.auto import tqdm

from ..games.bayesian import BayesianGame


class SCIPStatsHandler(pyscipopt.Eventhdlr):
    """PySCIPOpt Event handler to collect data for game value over time plots."""

    def __init__(self, progress_meter: tqdm | None = None):
        super().__init__()
        self.best_primal_bound: float = float("-inf")
        self.best_dual_bound: float = float("inf")
        self.nodes: list[tuple[float, float, float]] = []
        self.progress_meter = progress_meter

    def collectNodeInfo(self) -> None:
        pb = self.model.getPrimalbound()
        db = self.model.getDualbound()
        tm = self.model.getSolvingTime()
        change = False
        if pb > self.best_primal_bound:
            change = True
            self.best_primal_bound = pb
        if db < self.best_dual_bound:
            change = True
            self.best_dual_bound = db
        if change:
            self.nodes.append((pb, db, tm))

            if self.progress_meter is not None:
                self.progress_meter.set_description_str(f"MILP {pb} ... {db}")
                self.progress_meter.update(len(self.nodes))

    def eventinit(self) -> None:
        self.best_primal_bound = -self.model.infinity()
        self.best_dual_bound = self.model.infinity()
        self.model.catchEvent(pyscipopt.SCIP_EVENTTYPE.NODEINFEASIBLE, self)
        self.model.catchEvent(pyscipopt.SCIP_EVENTTYPE.NODEFEASIBLE, self)
        self.model.catchEvent(pyscipopt.SCIP_EVENTTYPE.NODEBRANCHED, self)

    def eventexec(self, _: pyscipopt.scip.Event) -> None:
        self.collectNodeInfo()

    def eventexit(self):
        self.model.dropEvent(pyscipopt.SCIP_EVENTTYPE.NODEINFEASIBLE, self)
        self.model.dropEvent(pyscipopt.SCIP_EVENTTYPE.NODEFEASIBLE, self)
        self.model.dropEvent(pyscipopt.SCIP_EVENTTYPE.NODEBRANCHED, self)


class DOBSS:
    class SCIPModel(pydantic.BaseModel):
        model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

        lp: pyscipopt.Model
        z: dict[str | int, dict[str | int, dict[str | int, pyscipopt.Variable]]]
        q: dict[str | int, dict[str | int, pyscipopt.Variable]]
        a: dict[str | int, pyscipopt.Variable]
        xi: pyscipopt.Variable
        l0: str | int

    class TimeSeries(pydantic.BaseModel):
        """
        Lower/upper value bound over time.
        """

        x: list[float]
        y: list[float]

    class Solution(pydantic.BaseModel):
        game_value: float
        defender_strategy: dict[str | int, float]
        attacker_strategy: dict[str | int, dict[str | int, float]]
        lower_bound: DOBSS.TimeSeries
        upper_bound: DOBSS.TimeSeries

    def __init__(
        self,
        game: BayesianGame,
        verbose: bool = False,
        presolve: bool = True,
        time_limit: float | None = None,
        random_seed: int | None = None,
    ):
        self.game = game
        self.verbose = verbose
        self.presolve = presolve
        self.time_limit = time_limit
        self.random_seed = random_seed
        self.solution = None
        self.bounds: list[tuple[float, float, float]] = []

    def milp_model(self) -> DOBSS.SCIPModel:
        lp = pyscipopt.Model()

        game = self.game.to_dicts()

        """Linearized terms z^l_ij = x_i q^l_j."""
        z = {
            t: {
                i: {
                    j: lp.addVar(f"z[{t}][{i}][{j}]", vtype="C", lb=0, ub=1)
                    for j in game.Q[t]
                }
                for i in game.X
            }
            for t in game.L
        }

        """Follower mixed strategy."""
        q = {
            t: {j: lp.addVar(f"q[{t}][{j}]", vtype="B") for j in game.Q[t]}
            for t in game.L
        }
        """Follower game value."""
        a = {t: lp.addVar(f"a[{t}]", vtype="C", lb=-lp.infinity()) for t in game.L}

        l0 = list(game.L)[0]

        for t in game.L:
            lp.addCons(
                pyscipopt.quicksum(z[t][i][j] for i in game.X for j in game.Q[t]) == 1
            )

        for t in game.L:
            for i in game.X:
                lp.addCons(pyscipopt.quicksum(z[t][i][j] for j in game.Q[t]) <= 1)

        for t in game.L:
            for j in game.Q[t]:
                lp.addCons(q[t][j] <= pyscipopt.quicksum(z[t][i][j] for i in game.X))
                lp.addCons(pyscipopt.quicksum(z[t][i][j] for i in game.X) <= 1)

        for t in game.L:
            lp.addCons(pyscipopt.quicksum(q[t][j] for j in game.Q[t]) == 1)

        for t in game.L:
            for j in game.Q[t]:
                lp.addCons(
                    a[t]
                    >= pyscipopt.quicksum(
                        game.C[t][i][j] * z[t][i][h] for h in game.Q[t] for i in game.X
                    )
                )
                lp.addCons(
                    a[t]
                    - pyscipopt.quicksum(
                        game.C[t][i][j] * z[t][i][h] for h in game.Q[t] for i in game.X
                    )
                    <= (1 - q[t][j]) * 100000
                )

        for t in game.L:
            for i in game.X:
                lp.addCons(
                    pyscipopt.quicksum(z[t][i][j] for j in game.Q[t])
                    == pyscipopt.quicksum(z[l0][i][j] for j in game.Q[l0])
                )

        obj = pyscipopt.quicksum(
            game.p[t] * game.R[t][i][j] * z[t][i][j]
            for t in game.L
            for i in game.X
            for j in game.Q[t]
        )
        xi = lp.addVar("xi", vtype="C", lb=-lp.infinity())
        lp.addCons(xi == obj)
        lp.setObjective(xi)
        lp.setMaximize()

        if not self.presolve:
            lp.setPresolve(pyscipopt.SCIP_PARAMSETTING.OFF)
            lp.setParam("misc/usesymmetry", "0")
        else:
            lp.setParam("presolving/maxrounds", 16)

        lp.setParam("estimation/restarts/restartpolicy", "n")
        lp.setParam("presolving/maxrestarts", 0)

        if self.time_limit:
            lp.setParam("limits/time", self.time_limit)

        if not self.verbose:
            lp.hideOutput(True)

        return DOBSS.SCIPModel(lp=lp, z=z, q=q, a=a, xi=xi, l0=l0)

    def solve(self) -> DOBSS.Solution:
        model = self.milp_model()

        with tqdm() as progress_meter:
            event_handler = SCIPStatsHandler(progress_meter)
            model.lp.includeEventhdlr(
                event_handler, "LPstat", "generate LP statistics after every LP event"
            )

            model.lp.optimize()

        if model.lp.getStage() == pyscipopt.SCIP_STAGE.PRESOLVING:
            event_handler.eventfree()
            model.lp.freeProb()
            del model.lp

            return DOBSS.Solution(
                game_value=float("-inf"),
                defender_strategy={},
                attacker_strategy={},
                lower_bound=DOBSS.TimeSeries(x=[], y=[]),
                upper_bound=DOBSS.TimeSeries(x=[], y=[]),
            )

        game = self.game.to_dicts()

        lower_bound = DOBSS.TimeSeries(
            x=[t for _, _, t in event_handler.nodes],
            y=[pb for pb, _, _ in event_handler.nodes],
        )
        upper_bound = DOBSS.TimeSeries(
            x=[t for _, _, t in event_handler.nodes],
            y=[db for _, db, _ in event_handler.nodes],
        )

        solution = DOBSS.Solution(
            game_value=model.lp.getVal(model.xi),
            defender_strategy={
                i: sum(
                    model.lp.getVal(model.z[model.l0][i][j]) for j in game.Q[model.l0]
                )
                for i in game.X
            },
            attacker_strategy={
                t: {j: model.lp.getVal(model.q[t][j]) for j in game.Q[t]}
                for t in game.L
            },
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        )

        event_handler.eventfree()
        model.lp.freeProb()
        del model.lp

        return solution

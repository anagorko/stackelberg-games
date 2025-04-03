from .problem import PatrollingProblem
from .setting import port_gdynia
from .space import PathSpace, PatrollingSpace, TensorProductSpace, WeightedGraphSpace


def gdynia_problem(edge_len_unit : int = 100, number_of_units: int = 1, observation_length : int = 1, att_time : int = 1) -> PatrollingProblem:
    environment = port_gdynia(number_of_units)

    base = PatrollingSpace(
        factors=[WeightedGraphSpace(graph=environment.topology[u], d=environment.length[u]) for u in environment.units],
        factor_coverage=[environment.coverage[u] for u in environment.units],
        targets=environment.targets,
        reward=environment.reward)

    # It might take longer time to attack targets that are further away from the port entrance
    tau = {t: att_time #+ min([nx.shortest_path_length(g, entrance, t, weight='len') for entrance in ['Aw_in1', 'Aw_in2']])
           for t in base.targets}
    horizon = max(tau.values()) + max(observation_length, 1)
    strategy = TensorProductSpace(factors=[PathSpace(base_space=base, length=horizon)])
    projection = {state: state[0][-1] for state in strategy.topology}

    return PatrollingProblem(
        base = base,
        strategy = strategy,
        projection = projection,
        targets = environment.targets,
        tau = {t: tau[t] for t in environment.targets},
        reward = environment.reward,
        observation_length = observation_length)
"""
This module implements defense plans.
"""

import networkx as nx
import pydantic
import random
from tqdm import tqdm

from .problem import PatrollingProblem
from .setting import Rational
from .schedule import Schedule
from .space import PathSpace, State, TensorProductSpace


class DefensePlan(pydantic.BaseModel):
    """
    Data for schedule generation.
    """

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    data: PatrollingProblem = pydantic.Field(exclude=True)
    """Problem specification."""

    transition: dict[State, dict[State, float]]
    """Transition probabilities. A map from edges to probabilities, 
    structured as [V -> [V -> [0, 1]]]."""

    stationary: dict[State, float]
    """Stationary distribution."""

    lower_bound: Rational
    """Lower bound on expected payoff, verified by solving LP formulation."""

    upper_bound: Rational
    """Upper bound on expected payoff, verified by solving LP formulation."""

    def generate_schedule(self, steps: int,
                          starting_state: State = None,
                          random_seed: int = None) -> Schedule:
        """Generates a schedule."""
        if random_seed is not None:
            random.seed(random_seed)

        if starting_state is None:
            starting_state = random.choices(list(self.stationary.keys()),
                                            weights=list(self.stationary.values()))[0]

        rollout = [starting_state]
        while len(rollout) < steps:
            current_state = rollout[-1]
            next_state = random.choices(list(self.transition[current_state].keys()),
                                        weights=list(self.transition[current_state].values()))[0]
            rollout.append(next_state)
        return Schedule(
            data=self.data,
            schedule=[self.data.projection[state] for state in rollout]
        )

    def monte_carlo_expected_reward(self, observation_len: int, rollout_len: int = 10**4, rollouts_num: int = 100,
                                    disable_tqdm: bool = False, print_debug: bool = False) -> float:
        """Monte Carlo evaluation of the defense plan against an opponent with a given observation length."""
        reward_history = {target: {} for target in self.data.targets }
        total_reward = {}
        for _ in tqdm(range(rollouts_num), disable=disable_tqdm):
            rollout = self.generate_schedule(rollout_len)
            reward = rollout.reward(observation_len)
            for observation, j in reward:
                tr, tc = total_reward.get((observation, j), (0, 0))
                total_reward[observation, j] = (tr + reward[observation, j][0],
                                                tc + reward[observation, j][1])

        min_reward = {target: float('inf') for target in self.data.targets }
        for observation, target in total_reward:
            avg_reward = total_reward[observation, target][0] / total_reward[observation, target][1]
            min_reward[target] = min(min_reward[target], avg_reward)

        if print_debug:
            for observation, target in total_reward:
                print('Rewards for attacking target', target)
                print('\t', observation, total_reward[observation, target][0] / total_reward[observation, target][1])
        return min(min_reward.values())
    
    def transition_graph(self) -> nx.DiGraph:
        tg = self.data.strategy.topology.copy()
        for v, prob in self.stationary.items():
            if prob == 0:
                tg.remove_node(v)
        tg.remove_edges_from([(v,w) for (v,w) in tg.edges if self.transition[v][w] == 0])
        return tg

    def is_irreducible(self) -> bool:
        return nx.is_strongly_connected(self.transition_graph())

    def force_strong_connectivity(self, connector_weight: float = None):
        if self.is_irreducible():
            return
        if connector_weight is None:
            connector_weight = min([v for t in self.transition.values() for v in t.values() if v > 0 ])
        sg = self.data.strategy.topology
        tg = self.transition_graph()
        components = list(nx.strongly_connected_components(tg))
        for c1 in components:
            for c2 in components:
                if c1 != c2:
                    (source, target) = min([(v1,v2) for v1 in c1 for v2 in c2],
                                    key = lambda pair: nx.shortest_path_length(sg, pair[0], pair[1]))
                    sp = nx.shortest_path(sg, source, target)
                    last = sp[0]
                    for v in sp[1:]:
                        self.transition[last][v] = max(self.transition[last][v], connector_weight)
                        last = v

    def project_to_sound(self, observation_len: int):
        """Projects the plan to a shorter observation length in order to achieve soundness."""
        
        def trim(state):
            s = list(state)
            s[0] = s[0][-observation_len:]
            return tuple(s)

        bags = { trim(long_state) : [] for long_state in self.stationary.keys()}
        for long_state in self.stationary.keys():
            bags[trim(long_state)].append(long_state)
        
        sound_stationary = { short_state : sum([self.stationary[s] for s in long_states])
                            for short_state, long_states in bags.items() }
        sound_transition = { short_state : {} for short_state in bags.keys() }
        for long_source, distribution in self.transition.items():
            short_source = trim(long_source)
            for long_dest, prob in distribution.items():
                short_dest = trim(long_dest)
                if short_dest not in sound_transition[short_source]:
                     sound_transition[short_source][short_dest] = 0.
                if prob > 0 and self.stationary[long_source] > 0:
                    sound_transition[short_source][short_dest] += prob * self.stationary[long_source] / sound_stationary[short_source]

        sound_strategy = TensorProductSpace(factors=[
                            PathSpace( base_space=self.data.base, length=observation_len)])
        return DefensePlan(
                data = PatrollingProblem(
                    base = self.data.base,
                    strategy = sound_strategy,
                    projection = {state: state[0][-1] for state in sound_strategy.topology},
                    targets = self.data.targets,
                    tau = self.data.tau,
                    reward = self.data.reward,
                    observation_length = observation_len
                ),
                stationary = sound_stationary,
                transition = sound_transition,
                lower_bound = 0,
                upper_bound = 0)
        

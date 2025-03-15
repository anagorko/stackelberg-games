"""
Unit tests for get_metadata / from_metadata methods.
"""

import itertools

import numpy as np

from stackelberg_games import core  # type: ignore[import-not-found]


def game_instances_normal_form() -> list[core.Game]:
    rng = np.random.default_rng()

    instances = []

    for m in range(1, 10):
        for n in range(1, 10):
            defender_payoff = rng.random(size=(m, n))
            attacker_payoff = rng.random(size=(m, n))

            defender_moves = [chr(ord("A") + i) for i in range(m)]
            attacker_moves = [chr(ord("a") + i) for i in range(n)]

            instances.append(core.NormalFormGame(defender_payoff, attacker_payoff))
            instances.append(
                core.NormalFormGame(
                    defender_payoff, attacker_payoff, defender_moves, attacker_moves
                )
            )

    return instances


def game_instances_bayesian() -> list[core.Game]:
    rng = np.random.default_rng()

    instances = []

    for m in range(1, 5):
        for n in range(1, 5):
            for t in range(1, 4):
                defender_payoff = [rng.random(size=(m, n)) for _ in range(t)]
                attacker_payoff = [rng.random(size=(m, n)) for _ in range(t)]

                defender_moves = [chr(ord("A") + i) for i in range(m)]
                attacker_moves = [
                    [chr(ord("a") + i) for i in range(n)] for _ in range(t)
                ]

                attacker_distribution = rng.random(size=(t,))

                instances.append(
                    core.BayesianGame(
                        defender_payoff, attacker_payoff, attacker_distribution
                    )
                )
                instances.append(
                    core.BayesianGame(
                        defender_payoff,
                        attacker_payoff,
                        attacker_distribution,
                        defender_moves,
                        attacker_moves,
                    )
                )

    return instances


def check_game_serialization(instance: core.Game) -> None:
    deserialized = instance.__class__.from_metadata(instance.get_metadata())
    # noinspection PyArgumentList
    reconstructed = instance.__class__(**instance.get_metadata().model_dump())

    assert deserialized == reconstructed == instance


def test_game_serialization() -> None:
    for instance in game_instances_normal_form() + game_instances_bayesian():
        check_game_serialization(instance)


def test_generator_serialization() -> None:
    generators = [
        core.RandomIndependentGameGenerator(m, n, t)
        for m in range(1, 5)
        for n in range(1, 5)
        for t in range(1, 5)
    ] + [
        core.RandomCorrelatedGameGenerator(m, n, t)
        for m in range(2, 5)
        for n in range(2, 5)
        for t in range(1, 5)
    ]

    for generator in generators:
        generator_metadata = generator.get_metadata()

        assert (
            generator.__class__(*generator_metadata.args, **generator_metadata.kwargs)
            == generator
        )
        assert generator.__class__.from_metadata(generator_metadata) == generator

        for random_seed in range(50):
            instance = generator.get_instance(random_seed)
            check_game_serialization(instance)

        for instance in itertools.islice(generator.get_series(random_seed=0), 50):
            check_game_serialization(instance)

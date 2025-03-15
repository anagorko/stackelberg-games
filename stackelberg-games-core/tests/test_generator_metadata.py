"""
Unit tests for GeneratorMetadata class.
"""

from stackelberg_games import core  # type: ignore[import-not-found]


def test_generator_metadata() -> None:
    metadata = core.GameGeneratorMetadata(
        cls="stackelberg_games.core.RandomIndependentGameGenerator",
        args=(),
        kwargs={
            "number_of_defender_moves": 128,
            "number_of_attacker_moves": 256,
            "number_of_attacker_types": 512,
        },
    )

    generator = core.GameGenerator.from_metadata(metadata)
    instance = generator.get_instance(0)

    metadata = core.GameGeneratorMetadata(
        cls="stackelberg_games.core.RandomIndependentGameGenerator",
        kwargs={
            "number_of_defender_moves": 128,
            "number_of_attacker_moves": 256,
            "number_of_attacker_types": 512,
        },
    )

    assert core.GameGenerator.from_metadata(metadata).get_instance(0) == instance

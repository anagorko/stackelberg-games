"""
Generator protocols.
"""

from __future__ import annotations

from collections.abc import Iterator
import sys
from typing import Any, Protocol, runtime_checkable

import numpy as np
import pydantic

from ..games.game import Game


class GameGeneratorMetadata(pydantic.BaseModel):
    cls: type[GameGenerator]
    args: tuple[Any, ...] = ()
    kwargs: dict[str, Any] = {}

    # noinspection PyNestedDecorators
    @pydantic.field_validator("cls", mode="before")
    @classmethod
    def class_from_string(
        cls, class_name: str | type[GameGenerator]
    ) -> type[GameGenerator]:
        if isinstance(class_name, GameGenerator):
            return class_name

        assert isinstance(class_name, str)

        module_name = ".".join(class_name.split(".")[:-1])
        class_name = class_name.split(".")[-1]
        return getattr(sys.modules[module_name], class_name)


@runtime_checkable
class GameGenerator(Protocol):
    def get_metadata(self) -> GameGeneratorMetadata:
        """Serializes the Generator instance to a pydantic.BaseModel."""

    @classmethod
    def from_metadata(cls, model: GameGeneratorMetadata) -> GameGenerator:
        """Deserializes a Generator instance from a pydantic.BaseModel."""

        return model.cls(*model.args, **model.kwargs)

    def get_instance(self, random_seed: int) -> Game:
        """Creates an instance of a game."""

    def get_series(self, random_seed: int) -> Iterator[Game]:
        """Creates a sequence of game instances."""

        rng = np.random.default_rng(random_seed)

        def next_instance() -> Iterator[Game]:
            while True:
                yield self.get_instance(rng.integers(0, 10**9))

        return next_instance()

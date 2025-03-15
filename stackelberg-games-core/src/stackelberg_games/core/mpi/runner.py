"""
This module implements distributed computation runner.
"""

from collections.abc import Iterator


class Task:
    cls: str


class Runner:
    def __init__(self, tasks: Iterator[Task]):
        pass

    def main(self) -> None:
        pass

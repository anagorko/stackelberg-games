from __future__ import annotations
from fractions import Fraction

import bisect
import matplotlib.pyplot as plt
import numpy


class PWLMap:
    """
    Piecewise-linear map on R.
    """

    def __init__(self, data: dict[int | Fraction, int | Fraction]):
        """Sample values, to be interpolated linearly."""

        self.data = data
        self.breakpoints = list(sorted(data.keys()))

    def __call__(self, x: int | Fraction) -> int | Fraction:
        """Compute value at x."""
        idx = bisect.bisect(self.breakpoints, x)
        if idx == 0:
            # x < a
            return self.data[self.breakpoints[0]]

        if self.breakpoints[idx-1] == x:
            return self.data[self.breakpoints[idx-1]]

        if idx == len(self.breakpoints):
            # x > b
            return self.data[self.breakpoints[-1]]

        left_argument = self.breakpoints[idx - 1]
        right_argument = self.breakpoints[idx]
        slope = Fraction(self.data[left_argument] - self.data[right_argument], left_argument - right_argument)

        return self.data[left_argument] + (x - left_argument) * slope

    def plot(self, ax: plt.Axes):
        """Plot self on matplotlib ax."""

        x = numpy.array(self.breakpoints)
        y = numpy.vectorize(self, otypes=[float])(x)
        ax.plot(x, y)

    def __add__(self, other: PWLMap):
        """Compute sum of two PWL maps."""

        data = {}
        for x in self.breakpoints + other.breakpoints:
            data[x] = self(x) + other(x)
        return PWLMap(data)

    def max(self, other: PWLMap):
        """Compute max of two PWL maps.

        Too bad PEP 8 forbids creation of our own dunder functions.
        """

        midpoints = []
        for a, b in zip(sorted(self.breakpoints + other.breakpoints), sorted(self.breakpoints + other.breakpoints)[1:]):
            if self(a) > other(a) and self(b) < other(b) or self(a) < other(a) and self(b) > other(b):
                midpoints.append(a + Fraction(other(a) - self(a), self(b) - self(a) + other(a) - other(b)) * (b - a))

        data = {}
        for x in self.breakpoints + other.breakpoints + midpoints:
            data[x] = max(self(x), other(x))
        return PWLMap(data)


def main():
    fig, ax = plt.subplots(2, 2, figsize=(6, 4), dpi=300)

    f = PWLMap({0: 4, 1: -1, 2: 0})
    g = PWLMap({0: -1, 1: 1, 2: 2})
    f.plot(ax[0, 0])
    g.plot(ax[1, 0])
    (f+g).plot(ax[0, 1])
    f.plot(ax[1, 1])
    g.plot(ax[1, 1])
    f.max(g).plot(ax[1, 1])

    plt.show()


if __name__ == '__main__':
    main()

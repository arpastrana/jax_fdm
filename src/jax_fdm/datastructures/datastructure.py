"""
A force density datastructure.
"""

from collections.abc import Iterator
from collections.abc import Sequence
from math import fabs
from statistics import stdev


class FDDatastructure:
    """
    A force density datastructure.

    This is a mixin of force-density-specific methods layered onto a concrete
    COMPAS datastructure. ``FDNetwork`` and ``FDMesh`` reach ``Datastructure``
    through ``Network`` and ``Mesh`` respectively, so this class does not
    inherit from it to avoid a redundant inheritance diamond.
    """

    # ----------------------------------------------------------------------
    # Edges
    # ----------------------------------------------------------------------

    def edge_load(
        self,
        key: tuple[int, int],
        load: list[float] | None = None,
    ) -> list[float] | None:
        """
        Gets or sets a load on an edge.
        """
        return self.edge_attributes(key, names=("px", "py", "pz"), values=load)

    def edge_forcedensity(
        self,
        key: tuple[int, int],
        q: float | None = None,
    ) -> float | None:
        """
        Gets or sets the force density on a single edge.
        """
        return self.edge_attribute(key, name="q", value=q)

    def edge_force(self, key: tuple[int, int]) -> float:
        """
        Gets the forces at a single edge the network.
        """
        return self.edge_attribute(key, name="force")

    def edge_loadpath(self, key: tuple[int, int]) -> float:
        """
        Gets the load path at a single edge the network.
        """
        force = self.edge_force(key)
        length = self.edge_attribute(key, name="length")
        return fabs(force * length)

    def edges_forcedensities(
        self,
        q: list[float] | None = None,
        keys: Sequence[tuple[int, int]] | None = None,
    ) -> list[float] | None:
        """
        Gets or sets the force densities on a list of edges.
        """
        return self.edges_attribute(name="q", value=q, keys=keys)

    def edges_forces(
        self,
        keys: Sequence[tuple[int, int]] | None = None,
    ) -> list[float]:
        """
        Gets the forces on the edges of the network.
        """
        return self.edges_attribute(keys=keys, name="force")

    def edges_lengths(
        self,
        keys: Sequence[tuple[int, int]] | None = None,
    ) -> list[float]:
        """
        Gets the lengths on the edges of the network.
        """
        return self.edges_attribute(keys=keys, name="length")

    def edges_loads(
        self,
        load: list[float] | None = None,
        keys: Sequence[tuple[int, int]] | None = None,
    ) -> list[list[float]] | None:
        """
        Gets or sets a load to the edges of the datastructure.
        """
        return self.edges_attributes(names=("px", "py", "pz"), values=load, keys=keys)

    def edges_loadpaths(
        self,
        keys: Sequence[tuple[int, int]] | None = None,
    ) -> Iterator[float]:
        """
        Gets the load path on the edges of the network.
        """
        keys = keys or self.edges()
        for key in keys:
            yield self.edge_loadpath(key)

    # ----------------------------------------------------------------------
    # Datastructure properties
    # ----------------------------------------------------------------------

    def loadpath(self) -> float:
        """
        Gets the total load path of the network.
        """
        return sum(list(self.edges_loadpaths()))

    def print_stats(
        self,
        other_stats: dict[str, list[float]] | None = None,
        ndigits: int = 3,
    ) -> None:
        """
        Print information aboud the equilibrium state of the network.
        """
        edges_pos = []
        edges_neg = []
        for edge in self.edges():
            _edges = edges_neg
            # getter-mode call always returns float
            if self.edge_forcedensity(edge) > 0.0:  # pyright: ignore[reportOptionalOperand]
                _edges = edges_pos
            _edges.append(edge)

        has_edges_pos = len(edges_pos) > 0
        has_edges_neg = len(edges_neg) > 0

        stats = {}
        if has_edges_neg:
            stats["FDs [-]"] = self.edges_forcedensities(keys=edges_neg)
        if has_edges_pos:
            stats["FDs [+]"] = self.edges_forcedensities(keys=edges_pos)
        if has_edges_neg:
            stats["Forces [-]"] = self.edges_forces(keys=edges_neg)
        if has_edges_pos:
            stats["Forces [+]"] = self.edges_forces(keys=edges_pos)

        stats["Lengths"] = self.edges_lengths()

        other_stats = other_stats or {}
        stats.update(other_stats)

        print(f"\n***{self.__class__.__name__} stats***")
        print(f"Load path: {round(self.loadpath(), ndigits)}")

        for name, vals in stats.items():
            if not vals:
                continue

            minv = round(min(vals), ndigits)
            maxv = round(max(vals), ndigits)
            meanv = round(sum(vals) / len(vals), ndigits)
            stdv = vals[0]
            if len(vals) > 1:
                stdv = round(stdev(vals), ndigits)

            name = f"{name:<18}"
            print(f"{name}\tMin: {minv}\tMax: {maxv}\tMean: {meanv}\tStDev: {stdv}")

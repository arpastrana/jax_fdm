"""A force density datastructure."""

from collections.abc import Iterable
from collections.abc import Iterator
from math import fabs
from statistics import stdev

from jax_fdm.datastructures.types import FDDatastructureType


class FDDatastructure(FDDatastructureType):
    """
    A force density datastructure.

    This is a mixin of force-density-specific methods layered onto a concrete
    COMPAS datastructure. ``FDNetwork`` and ``FDMesh`` reach ``Datastructure``
    through ``Network`` and ``Mesh`` respectively, so this class does not
    inherit from it to avoid a redundant inheritance diamond. The typing-only
    base declares the COMPAS accessors this mixin calls.
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
        Get or set the load vector on a single edge.

        Parameters
        ----------
        key :
            The edge to access.
        load :
            The load vector to set. If None, the current load is returned.

        Returns
        -------
        load :
            The edge's load vector.
        """
        return self.edge_attributes(key, names=("px", "py", "pz"), values=load)

    def edge_forcedensity(
        self,
        key: tuple[int, int],
        q: float | None = None,
    ) -> float | None:
        """
        Get or set the force density on a single edge.

        Parameters
        ----------
        key :
            The edge to access.
        q :
            The force density to set. If None, the current value is returned.

        Returns
        -------
        q :
            The edge's force density.
        """
        return self.edge_attribute(key, name="q", value=q)

    def edge_force(self, key: tuple[int, int]) -> float:
        """
        Get the internal force in a single edge.

        Parameters
        ----------
        key :
            The edge to access.

        Returns
        -------
        force :
            The edge's internal force.
        """
        return self.edge_attribute(key, name="force")

    def edge_loadpath(self, key: tuple[int, int]) -> float:
        """
        Get the load path of a single edge.

        Parameters
        ----------
        key :
            The edge to access.

        Returns
        -------
        loadpath :
            The absolute product of the edge's force and length.
        """
        force = self.edge_force(key)
        length = self.edge_attribute(key, name="length")
        return fabs(force * length)

    def edges_forcedensities(
        self,
        q: float | None = None,
        keys: Iterable[tuple[int, int]] | None = None,
    ) -> list[float] | None:
        """
        Get or set the force densities on many edges.

        Parameters
        ----------
        q :
            The force density to set on every edge. If None, the current values
            are returned.
        keys :
            The edges to access. If None, all edges are used.

        Returns
        -------
        q :
            The force density of each edge.
        """
        return self.edges_attribute(name="q", value=q, keys=keys)

    def edges_forces(
        self,
        keys: Iterable[tuple[int, int]] | None = None,
    ) -> list[float]:
        """
        Get the internal forces of many edges.

        Parameters
        ----------
        keys :
            The edges to access. If None, all edges are used.

        Returns
        -------
        forces :
            The internal force of each edge.
        """
        return self.edges_attribute(keys=keys, name="force")

    def edges_lengths(
        self,
        keys: Iterable[tuple[int, int]] | None = None,
    ) -> list[float]:
        """
        Get the lengths of many edges.

        Parameters
        ----------
        keys :
            The edges to access. If None, all edges are used.

        Returns
        -------
        lengths :
            The length of each edge.
        """
        return self.edges_attribute(keys=keys, name="length")

    def edges_loads(
        self,
        load: list[float] | None = None,
        keys: Iterable[tuple[int, int]] | None = None,
    ) -> list[list[float]] | None:
        """
        Get or set the load vectors on many edges.

        Parameters
        ----------
        load :
            The load vector to set on each edge. If None, current loads are returned.
        keys :
            The edges to access. If None, all edges are used.

        Returns
        -------
        loads :
            The load vector of each edge.
        """
        return self.edges_attributes(names=("px", "py", "pz"), values=load, keys=keys)

    def edges_loadpaths(
        self,
        keys: Iterable[tuple[int, int]] | None = None,
    ) -> Iterator[float]:
        """
        Iterate over the load path of many edges.

        Parameters
        ----------
        keys :
            The edges to access. If None, all edges are used.

        Yields
        ------
        loadpath :
            The load path of each edge.
        """
        edges = self.edges() if keys is None else keys
        for key in edges:
            yield self.edge_loadpath(key)

    # ----------------------------------------------------------------------
    # Datastructure properties
    # ----------------------------------------------------------------------

    def loadpath(self) -> float:
        """
        Get the total load path summed over all edges.

        Returns
        -------
        loadpath :
            The sum of the per-edge load paths.
        """
        return sum(list(self.edges_loadpaths()))

    def print_stats(
        self,
        other_stats: dict[str, list[float]] | None = None,
        ndigits: int = 3,
    ) -> None:
        """
        Print summary statistics of the datastructure's equilibrium state.

        Parameters
        ----------
        other_stats :
            Extra named value lists to summarize alongside the built-in ones.
        ndigits :
            The number of digits to round the printed statistics to.
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

"""
A force density network.
"""

from collections.abc import Iterator
from typing import Any

from compas.datastructures import Mesh
from compas.datastructures import Network
from jax_fdm.datastructures import FDDatastructure


class FDNetwork(Network, FDDatastructure):
    """
    A force density network.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        self.update_default_edge_attributes(
            {"q": 0.0, "length": 0.0, "force": 0.0, "px": 0.0, "py": 0.0, "pz": 0.0},
        )

        self.update_default_node_attributes(
            {
                "x": 0.0,
                "y": 0.0,
                "z": 0.0,
                "px": 0.0,
                "py": 0.0,
                "pz": 0.0,
                "rx": 0.0,
                "ry": 0.0,
                "rz": 0.0,
                "is_support": False,
            },
        )

    # ----------------------------------------------------------------------
    # Constructors
    # ----------------------------------------------------------------------

    @classmethod
    def from_mesh(cls, mesh: Mesh) -> "FDNetwork":
        """
        Create a force density network from a mesh.
        """
        nodes = {vkey: mesh.vertex_coordinates(vkey) for vkey in mesh.vertices()}
        network = cls.from_nodes_and_edges(nodes, mesh.edges())

        for node in network.nodes():
            attrs = mesh.vertex_attributes(node)
            # names=None getter always returns a dict-like view, not None
            network.node_attributes(node, names=attrs.keys(), values=attrs.values())  # pyright: ignore[reportOptionalMemberAccess]

        for edge in network.edges():
            attrs = mesh.edge_attributes(edge)
            # names=None getter always returns a dict-like view, not None
            network.edge_attributes(edge, names=attrs.keys(), values=attrs.values())  # pyright: ignore[reportOptionalMemberAccess]

        return network

    # ----------------------------------------------------------------------
    # Nodes
    # ----------------------------------------------------------------------

    def nodes_coordinates(
        self,
        keys: list[int] | None = None,
        axes: str = "xyz",
    ) -> list[list[float]]:
        """
        Gets or sets the x, y, z coordinates of a list of nodes.
        """
        node_keys = keys or self.nodes()
        return [self.node_coordinates(node, axes) for node in node_keys]

    def nodes_fixedcoordinates(
        self,
        keys: list[int] | None = None,
        axes: str = "xyz",
    ) -> list[list[float]]:
        """
        Gets the x, y, z coordinates of the anchors of the network.
        """
        if keys:
            node_keys = {key for key in keys if self.is_node_support(key)}
        else:
            node_keys = self.nodes_fixed()

        # nodes_fixed() with no keys always returns a generator, never None
        return [self.node_coordinates(node, axes) for node in node_keys]  # pyright: ignore[reportOptionalIterable]

    def number_of_anchors(self) -> int:
        """
        The number of anchored nodes.
        """
        # nodes_anchors() with no keys always returns a generator, never None
        return len(list(self.nodes_anchors()))  # pyright: ignore[reportArgumentType]

    def number_of_supports(self) -> int:
        """
        The number of supported nodes.
        """
        # nodes_supports() with no keys always returns a generator, never None
        return len(list(self.nodes_supports()))  # pyright: ignore[reportArgumentType]

    def node_support(self, key: int) -> None:
        """
        Sets a node as a fixed anchor.
        """
        # setter-mode call always returns None
        return self.node_attribute(key=key, name="is_support", value=True)  # pyright: ignore[reportReturnType]

    def node_anchor(self, key: int) -> None:
        """
        Sets a node as a fixed anchor.
        """
        return self.node_support(key)

    def is_node_support(self, key: int) -> bool:
        """
        Test if the node is a fixed node.
        """
        # getter-mode call always returns the bool default
        return self.node_attribute(key=key, name="is_support")  # pyright: ignore[reportReturnType]

    def nodes_supports(self, keys: list[int] | None = None) -> Iterator[int] | None:
        """
        Gets or sets the node keys where a support has been assigned.
        """
        if keys is None:
            # data=False getter always yields plain node keys
            return self.nodes_where({"is_support": True})  # pyright: ignore[reportReturnType]

        # setter-mode call always returns None
        return self.nodes_attribute(name="is_support", value=True, keys=keys)  # pyright: ignore[reportReturnType]

    def nodes_fixed(self, keys: list[int] | None = None) -> Iterator[int] | None:
        """
        Gets or sets the node keys where a support has been assigned.
        """
        return self.nodes_supports(keys)

    def nodes_anchors(self, keys: list[int] | None = None) -> Iterator[int] | None:
        """
        Gets or sets the node keys where an anchor has been assigned.
        """
        return self.nodes_supports(keys)

    def nodes_free(self) -> Iterator[int]:
        """
        The keys of the nodes where there is no support assigned.
        """
        # data=False getter always yields plain node keys
        return self.nodes_where({"is_support": False})  # pyright: ignore[reportReturnType]

    def node_load(
        self,
        key: int,
        load: list[float] | None = None,
    ) -> list[float] | None:
        """
        Gets or sets a load to the nodes of the network.
        """
        # names given as a non-empty tuple always returns a list
        return self.node_attributes(key=key, names=("px", "py", "pz"), values=load)  # pyright: ignore[reportReturnType]

    def nodes_loads(
        self,
        load: list[float] | None = None,
        keys: list[int] | None = None,
    ) -> list[list[float]] | None:
        """
        Gets or sets a load to the nodes of the network.
        """
        # names given as a non-empty tuple always returns a list of lists
        return self.nodes_attributes(names=("px", "py", "pz"), values=load, keys=keys)  # pyright: ignore[reportReturnType]

    def nodes_residual(self, keys: list[int] | None = None) -> list[list[float]]:
        """
        Gets the residual forces of the nodes of the network.
        """
        # names given as a non-empty tuple always returns a list of lists
        return self.nodes_attributes(names=("rx", "ry", "rz"), keys=keys)  # pyright: ignore[reportReturnType]

    def node_residual(self, key: int) -> list[float]:
        """
        Gets the residual force of a single node of the network.
        """
        # names given as a non-empty tuple always returns a list
        return self.node_attributes(key=key, names=("rx", "ry", "rz"))  # pyright: ignore[reportReturnType]

    def nodes_reactions(self, keys: list[int] | None = None) -> list[list[float]]:
        """
        Gets the reaction forces of the nodes of the network.
        """
        # nodes_fixed() with no keys always returns a generator, never None
        keys = keys or self.nodes_fixed()  # pyright: ignore[reportAssignmentType]
        # names given as a non-empty tuple always returns a list of lists
        return self.nodes_attributes(names=("rx", "ry", "rz"), keys=keys)  # pyright: ignore[reportReturnType]

    def node_reaction(self, key: int) -> list[float]:
        """
        Gets the reaction force of a single node of the network.
        """
        # names given as a non-empty tuple always returns a list
        return self.node_attributes(key=key, names=("rx", "ry", "rz"))  # pyright: ignore[reportReturnType]

    # ----------------------------------------------------------------------
    # Edges
    # ----------------------------------------------------------------------

    def edges(self, data: bool = False) -> Iterator[tuple[int, int]]:
        """
        Iterate over the edges of the network.
        """
        # data=False getter always yields plain (u, v) edge keys
        return super().edges(data)  # pyright: ignore[reportReturnType]

    def is_edge_supported(self, key: tuple[int, int]) -> bool:
        """
        Test if any of the two nodes connected by the edge is a support.
        """
        return any(self.is_node_support(node) for node in key)

    def is_edge_fully_supported(self, key: tuple[int, int]) -> bool:
        """
        Test if the two nodes connected the edge are a support.
        """
        return all(self.is_node_support(node) for node in key)

    # ----------------------------------------------------------------------
    # Datastructure properties
    # ----------------------------------------------------------------------

    def parameters(self) -> tuple[list[float], list[list[float]], list[list[float]]]:
        """
        Return the design parameters of the network.
        """
        q = self.edges_forcedensities()
        xyz_fixed = self.nodes_fixedcoordinates()
        loads = self.nodes_loads()

        # getter-mode calls always return lists, never None
        assert q is not None
        assert loads is not None

        return q, xyz_fixed, loads

"""A force density network."""

from collections.abc import Iterable
from collections.abc import Iterator
from typing import Any
from typing import Self
from typing import overload

from compas.datastructures import Mesh
from compas.datastructures import Network
from jax_fdm.datastructures import FDDatastructure
from jax_fdm.datastructures.types import FDNetworkType


class FDNetwork(FDNetworkType, Network, FDDatastructure):
    """
    A force density network.

    Notes
    -----
    The typing-only first base re-narrows the COMPAS constructors and accessors
    for static checkers; it is empty at runtime.
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
    def from_mesh(cls, mesh: Mesh) -> Self:
        """
        Build a force density network from a mesh's vertices and edges.

        Parameters
        ----------
        mesh :
            The mesh to copy vertices, edges, and their attributes from.

        Returns
        -------
        network :
            The network mirroring the mesh's connectivity and attributes.
        """
        nodes = {vkey: mesh.vertex_coordinates(vkey) for vkey in mesh.vertices()}
        network = cls.from_nodes_and_edges(nodes, mesh.edges())

        for node in network.nodes():
            # the names=None getter always returns a dict-like attribute view
            attrs = dict(mesh.vertex_attributes(node) or {})
            network.node_attributes(node, names=attrs.keys(), values=attrs.values())

        for edge in network.edges():
            # the names=None getter always returns a dict-like attribute view
            attrs = dict(mesh.edge_attributes(edge) or {})
            network.edge_attributes(edge, names=attrs.keys(), values=attrs.values())

        return network

    # ----------------------------------------------------------------------
    # Nodes
    # ----------------------------------------------------------------------

    def nodes_coordinates(
        self,
        keys: Iterable[int] | None = None,
        axes: str = "xyz",
    ) -> list[list[float]]:
        """
        Get the coordinates of many nodes.

        Parameters
        ----------
        keys :
            The nodes to access. If None, all nodes are used.
        axes :
            The coordinate axes to return, as a subset of ``"xyz"``.

        Returns
        -------
        coordinates :
            The selected coordinates of each node.
        """
        node_keys = keys or self.nodes()
        return [self.node_coordinates(node, axes) for node in node_keys]

    def nodes_fixedcoordinates(
        self,
        keys: Iterable[int] | None = None,
        axes: str = "xyz",
    ) -> list[list[float]]:
        """
        Get the coordinates of the supported nodes.

        Parameters
        ----------
        keys :
            The candidate nodes; only the supported ones are kept. If None, all
            supported nodes are used.
        axes :
            The coordinate axes to return, as a subset of ``"xyz"``.

        Returns
        -------
        coordinates :
            The selected coordinates of each supported node.
        """
        if keys:
            node_keys = {key for key in keys if self.is_node_support(key)}
        else:
            node_keys = self.nodes_fixed()

        return [self.node_coordinates(node, axes) for node in node_keys]

    def number_of_anchors(self) -> int:
        """
        The number of anchored nodes.
        """
        return len(list(self.nodes_anchors()))

    def number_of_supports(self) -> int:
        """
        The number of supported nodes.
        """
        return len(list(self.nodes_supports()))

    def node_support(self, key: int) -> None:
        """
        Mark a node as a support.

        Parameters
        ----------
        key :
            The node to fix.
        """
        # setter-mode call always returns None
        return self.node_attribute(key=key, name="is_support", value=True)

    def node_anchor(self, key: int) -> None:
        """
        Mark a node as a support.

        Parameters
        ----------
        key :
            The node to fix.

        Notes
        -----
        An alias of `node_support`; anchor and support are synonyms here.
        """
        return self.node_support(key)

    def is_node_support(self, key: int) -> bool:
        """
        Test whether a node is a support.

        Parameters
        ----------
        key :
            The node to test.

        Returns
        -------
        is_support :
            True if the node is a support.
        """
        # getter-mode call always returns the bool default
        return self.node_attribute(key=key, name="is_support")

    @overload
    def nodes_supports(self, keys: None = None) -> Iterator[int]: ...
    @overload
    def nodes_supports(self, keys: Iterable[int]) -> None: ...
    def nodes_supports(
        self,
        keys: Iterable[int] | None = None,
    ) -> Iterator[int] | None:
        """
        Get the support nodes, or mark nodes as supports.

        Parameters
        ----------
        keys :
            The nodes to mark as supports. If None, the existing support nodes are
            returned instead.

        Returns
        -------
        supports :
            The support node keys when reading; None when setting.
        """
        if keys is None:
            # data=False getter always yields plain node keys
            return self.nodes_where({"is_support": True})

        # setter-mode call always returns None
        return self.nodes_attribute(name="is_support", value=True, keys=keys)  # pyright: ignore[reportReturnType]

    @overload
    def nodes_fixed(self, keys: None = None) -> Iterator[int]: ...
    @overload
    def nodes_fixed(self, keys: Iterable[int]) -> None: ...
    def nodes_fixed(self, keys: Iterable[int] | None = None) -> Iterator[int] | None:
        """
        Get the support nodes, or mark nodes as supports.

        Parameters
        ----------
        keys :
            The nodes to mark as supports. If None, the existing support nodes are
            returned instead.

        Returns
        -------
        supports :
            The support node keys when reading; None when setting.

        Notes
        -----
        An alias of `nodes_supports`.
        """
        return self.nodes_supports(keys)

    @overload
    def nodes_anchors(self, keys: None = None) -> Iterator[int]: ...
    @overload
    def nodes_anchors(self, keys: Iterable[int]) -> None: ...
    def nodes_anchors(self, keys: Iterable[int] | None = None) -> Iterator[int] | None:
        """
        Get the support nodes, or mark nodes as supports.

        Parameters
        ----------
        keys :
            The nodes to mark as supports. If None, the existing support nodes are
            returned instead.

        Returns
        -------
        supports :
            The support node keys when reading; None when setting.

        Notes
        -----
        An alias of `nodes_supports`.
        """
        return self.nodes_supports(keys)

    def nodes_free(self) -> Iterator[int]:
        """
        Iterate over the free (unsupported) nodes.

        Returns
        -------
        nodes_free :
            The keys of the nodes that are not supports.
        """
        # data=False getter always yields plain node keys
        return self.nodes_where({"is_support": False})

    def node_load(
        self,
        key: int,
        load: list[float] | None = None,
    ) -> list[float] | None:
        """
        Get or set the load vector on a single node.

        Parameters
        ----------
        key :
            The node to access.
        load :
            The load vector to set. If None, the current load is returned.

        Returns
        -------
        load :
            The node's load vector.
        """
        # names given as a non-empty tuple always returns a list
        return self.node_attributes(key=key, names=("px", "py", "pz"), values=load)

    def nodes_loads(
        self,
        load: list[float] | None = None,
        keys: Iterable[int] | None = None,
    ) -> list[list[float]] | None:
        """
        Get or set the load vectors on many nodes.

        Parameters
        ----------
        load :
            The load vector to set on each node. If None, current loads are returned.
        keys :
            The nodes to access. If None, all nodes are used.

        Returns
        -------
        loads :
            The load vector of each node.
        """
        # names given as a non-empty tuple always returns a list of lists
        return self.nodes_attributes(names=("px", "py", "pz"), values=load, keys=keys)  # pyright: ignore[reportReturnType]

    def nodes_residual(self, keys: Iterable[int] | None = None) -> list[list[float]]:
        """
        Get the residual force vectors of many nodes.

        Parameters
        ----------
        keys :
            The nodes to access. If None, all nodes are used.

        Returns
        -------
        residuals :
            The residual force vector of each node.
        """
        # names given as a non-empty tuple always returns a list of lists
        return self.nodes_attributes(names=("rx", "ry", "rz"), keys=keys)  # pyright: ignore[reportReturnType]

    def node_residual(self, key: int) -> list[float]:
        """
        Get the residual force vector of a single node.

        Parameters
        ----------
        key :
            The node to access.

        Returns
        -------
        residual :
            The node's residual force vector.
        """
        # names given as a non-empty tuple always returns a list
        return self.node_attributes(key=key, names=("rx", "ry", "rz"))

    def nodes_reactions(self, keys: Iterable[int] | None = None) -> list[list[float]]:
        """
        Get the reaction force vectors of the support nodes.

        Parameters
        ----------
        keys :
            The nodes to access. If None, all support nodes are used.

        Returns
        -------
        reactions :
            The reaction force vector of each selected node.
        """
        # nodes_fixed() with no keys always returns a generator, never None
        keys = keys or self.nodes_fixed()
        # names given as a non-empty tuple always returns a list of lists
        return self.nodes_attributes(names=("rx", "ry", "rz"), keys=keys)  # pyright: ignore[reportReturnType]

    def node_reaction(self, key: int) -> list[float]:
        """
        Get the reaction force vector of a single node.

        Parameters
        ----------
        key :
            The node to access.

        Returns
        -------
        reaction :
            The node's reaction force vector.
        """
        # names given as a non-empty tuple always returns a list
        return self.node_attributes(key=key, names=("rx", "ry", "rz"))

    # ----------------------------------------------------------------------
    # Edges
    # ----------------------------------------------------------------------

    def is_edge_supported(self, key: tuple[int, int]) -> bool:
        """
        Test whether either end node of an edge is a support.

        Parameters
        ----------
        key :
            The edge to test.

        Returns
        -------
        is_supported :
            True if at least one of the edge's nodes is a support.
        """
        return any(self.is_node_support(node) for node in key)

    def is_edge_fully_supported(self, key: tuple[int, int]) -> bool:
        """
        Test whether both end nodes of an edge are supports.

        Parameters
        ----------
        key :
            The edge to test.

        Returns
        -------
        is_fully_supported :
            True if both of the edge's nodes are supports.
        """
        return all(self.is_node_support(node) for node in key)

    # ----------------------------------------------------------------------
    # Datastructure properties
    # ----------------------------------------------------------------------

    def parameters(self) -> tuple[list[float], list[list[float]], list[list[float]]]:
        """
        Return the force density design parameters of the network.

        Returns
        -------
        parameters :
            The edge force densities, the fixed node coordinates, and the node
            loads.
        """
        q = self.edges_forcedensities()
        xyz_fixed = self.nodes_fixedcoordinates()
        loads = self.nodes_loads()

        # getter-mode calls always return lists, never None
        assert q is not None
        assert loads is not None

        return q, xyz_fixed, loads

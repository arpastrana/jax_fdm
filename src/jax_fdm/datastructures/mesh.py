"""A force density mesh."""

from collections.abc import Iterable
from collections.abc import Iterator
from collections.abc import Sequence
from typing import Any
from typing import overload

import jax.numpy as jnp

from compas.datastructures import Mesh
from jax_fdm.datastructures.datastructure import FDDatastructure
from jax_fdm.datastructures.types import FDMeshType
from jax_fdm.geometry import polygon_lcs

__all__ = ["FDMesh"]


class FDMesh(FDMeshType, Mesh, FDDatastructure):
    """
    A force density mesh.

    Notes
    -----
    The typing-only first base re-narrows the COMPAS constructors and accessors
    for static checkers; it is empty at runtime.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        self.update_default_edge_attributes(self.edge_attributes_default)

        self.update_default_vertex_attributes(
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

        self.update_default_face_attributes({"px": 0.0, "py": 0.0, "pz": 0.0})

    # ----------------------------------------------------------------------
    # Node
    # ----------------------------------------------------------------------

    def is_vertex_support(self, key: int) -> bool:
        """
        Test whether a vertex is a support.

        Parameters
        ----------
        key :
            The vertex to test.

        Returns
        -------
        is_support :
            True if the vertex is a support.
        """
        # getter-mode call always returns the bool default
        return self.vertex_attribute(key, name="is_support")

    def number_of_supports(self) -> int:
        """
        Count the supported vertices.

        Returns
        -------
        count :
            The number of support vertices.
        """
        return len(list(self.vertices_supports()))

    def vertex_support(self, key: int) -> None:
        """
        Mark a vertex as a support.

        Parameters
        ----------
        key :
            The vertex to fix.
        """
        # setter-mode call always returns None
        return self.vertex_attribute(key, name="is_support", value=True)

    def vertex_load(
        self,
        key: int,
        load: Iterable[float] | None = None,
    ) -> list[float] | None:
        """
        Get or set the load vector on a single vertex.

        Parameters
        ----------
        key :
            The vertex to access.
        load :
            The load vector to set. If None, the current load is returned.

        Returns
        -------
        load :
            The vertex's load vector.
        """
        # names given as a non-empty tuple always returns a list
        return self.vertex_attributes(key, names=("px", "py", "pz"), values=load)

    def vertex_residual(self, key: int) -> list[float]:
        """
        Get the residual force vector of a single vertex.

        Parameters
        ----------
        key :
            The vertex to access.

        Returns
        -------
        residual :
            The vertex's residual force vector.
        """
        # names given as a non-empty tuple always returns a list
        return self.vertex_attributes(key, names=("rx", "ry", "rz"))

    def vertex_reaction(self, key: int) -> list[float]:
        """
        Get the reaction force vector of a single vertex.

        Parameters
        ----------
        key :
            The vertex to access.

        Returns
        -------
        reaction :
            The vertex's reaction force, equal to its residual.
        """
        return self.vertex_residual(key)

    def vertices_coordinates(
        self,
        keys: Iterable[int] | None = None,
        axes: str = "xyz",
    ) -> list[list[float]]:
        """
        Get the coordinates of many vertices.

        Parameters
        ----------
        keys :
            The vertices to access. If None, all vertices are used.
        axes :
            The coordinate axes to return, as a subset of ``"xyz"``.

        Returns
        -------
        coordinates :
            The selected coordinates of each vertex.
        """
        vertex_keys = keys or self.vertices()
        # inherited COMPAS getter always returns a coordinate list here
        return [self.vertex_coordinates(node, axes) for node in vertex_keys]

    def vertices_fixedcoordinates(
        self,
        keys: Iterable[int] | None = None,
        axes: str = "xyz",
    ) -> list[list[float]]:
        """
        Get the coordinates of the supported vertices.

        Parameters
        ----------
        keys :
            The candidate vertices; only the supported ones are kept. If None, all
            supported vertices are used.
        axes :
            The coordinate axes to return, as a subset of ``"xyz"``.

        Returns
        -------
        coordinates :
            The selected coordinates of each supported vertex.
        """
        if keys:
            vertex_keys = {key for key in keys if self.is_vertex_support(key)}
        else:
            vertex_keys = self.vertices_fixed()

        return [self.vertex_coordinates(node, axes) for node in vertex_keys]

    @overload
    def vertices_supports(self, keys: None = None) -> Iterator[int]: ...
    @overload
    def vertices_supports(self, keys: Iterable[int]) -> None: ...
    def vertices_supports(
        self,
        keys: Iterable[int] | None = None,
    ) -> Iterator[int] | None:
        """
        Get the support vertices, or mark vertices as supports.

        Parameters
        ----------
        keys :
            The vertices to mark as supports. If None, the existing support vertices
            are returned instead.

        Returns
        -------
        supports :
            The support vertex keys when reading; None when setting.
        """
        if keys is None:
            # data=False getter always yields plain vertex keys
            return self.vertices_where({"is_support": True})

        # setter-mode call always returns None
        return self.vertices_attribute(name="is_support", value=True, keys=keys)  # pyright: ignore[reportReturnType]

    @overload
    def vertices_fixed(self, keys: None = None) -> Iterator[int]: ...
    @overload
    def vertices_fixed(self, keys: Iterable[int]) -> None: ...
    def vertices_fixed(self, keys: Iterable[int] | None = None) -> Iterator[int] | None:
        """
        Get the support vertices, or mark vertices as supports.

        Parameters
        ----------
        keys :
            The vertices to mark as supports. If None, the existing support vertices
            are returned instead.

        Returns
        -------
        supports :
            The support vertex keys when reading; None when setting.

        Notes
        -----
        An alias of `vertices_supports`.
        """
        return self.vertices_supports(keys)

    def vertices_free(self) -> Iterator[int]:
        """
        Iterate over the free (unsupported) vertices.

        Returns
        -------
        vertices_free :
            The keys of the vertices that are not supports.
        """
        # data=False getter always yields plain vertex keys
        return self.vertices_where({"is_support": False})

    def vertices_loads(
        self,
        load: Sequence[float] | None = None,
        keys: Iterable[int] | None = None,
    ) -> list[list[float]] | None:
        """
        Get or set the load vectors on many vertices.

        Parameters
        ----------
        load :
            The load vector to set on each vertex. If None, current loads are
            returned.
        keys :
            The vertices to access. If None, all vertices are used.

        Returns
        -------
        loads :
            The load vector of each vertex.
        """
        # names given as a non-empty tuple always returns a list of lists
        return self.vertices_attributes(
            names=("px", "py", "pz"),
            values=load,
            keys=keys,
        )  # pyright: ignore[reportReturnType]

    def vertices_residual(self, keys: Iterable[int] | None = None) -> list[list[float]]:
        """
        Get the residual force vectors of many vertices.

        Parameters
        ----------
        keys :
            The vertices to access. If None, all vertices are used.

        Returns
        -------
        residuals :
            The residual force vector of each vertex.
        """
        # names given as a non-empty tuple always returns a list of lists
        return self.vertices_attributes(names=("rx", "ry", "rz"), keys=keys)  # pyright: ignore[reportReturnType]

    def vertices_reactions(
        self,
        keys: Iterable[int] | None = None,
    ) -> list[list[float]]:
        """
        Get the reaction force vectors of the support vertices.

        Parameters
        ----------
        keys :
            The vertices to access. If None, all support vertices are used.

        Returns
        -------
        reactions :
            The reaction force vector of each selected vertex.
        """
        # vertices_fixed() with no keys always returns a generator, never None
        keys = keys or self.vertices_fixed()
        return self.vertices_residual(keys)

    # ----------------------------------------------------------------------
    # Edges
    # ----------------------------------------------------------------------

    def is_edge_supported(self, key: tuple[int, int]) -> bool:
        """
        Test whether either end vertex of an edge is a support.

        Parameters
        ----------
        key :
            The edge to test.

        Returns
        -------
        is_supported :
            True if at least one of the edge's vertices is a support.
        """
        return any(self.is_vertex_support(vertex) for vertex in key)

    def is_edge_fully_supported(self, key: tuple[int, int]) -> bool:
        """
        Test whether both end vertices of an edge are supports.

        Parameters
        ----------
        key :
            The edge to test.

        Returns
        -------
        is_fully_supported :
            True if both of the edge's vertices are supports.
        """
        return all(self.is_vertex_support(vertex) for vertex in key)

    # ----------------------------------------------------------------------
    # Faces
    # ----------------------------------------------------------------------

    def face_lcs(self, key: int) -> list[list[float]]:
        """
        Compute the local coordinate frame of a face.

        Parameters
        ----------
        key :
            The face to access.

        Returns
        -------
        lcs :
            The three axes of the face's local coordinate system.
        """
        return polygon_lcs(jnp.asarray(self.face_coordinates(key))).tolist()

    def face_load(
        self,
        key: int,
        load: list[float] | None = None,
    ) -> list[float] | None:
        """
        Get or set the load vector on a single face.

        Parameters
        ----------
        key :
            The face to access.
        load :
            The load vector to set. If None, the current load is returned.

        Returns
        -------
        load :
            The face's load vector.
        """
        # names given as a non-empty tuple always returns a list
        return self.face_attributes(key=key, names=("px", "py", "pz"), values=load)  # pyright: ignore[reportReturnType]

    def is_face_supported(self, key: int) -> bool:
        """
        Test whether any vertex of a face is a support.

        Parameters
        ----------
        key :
            The face to test.

        Returns
        -------
        is_supported :
            True if at least one of the face's vertices is a support.
        """
        return any(self.is_vertex_support(vertex) for vertex in self.face_vertices(key))

    def is_face_fully_supported(self, key: int) -> bool:
        """
        Test whether every vertex of a face is a support.

        Parameters
        ----------
        key :
            The face to test.

        Returns
        -------
        is_fully_supported :
            True if all of the face's vertices are supports.
        """
        return all(self.is_vertex_support(vertex) for vertex in self.face_vertices(key))

    def faces_loads(
        self,
        load: list[float] | None = None,
        keys: Iterable[int] | None = None,
    ) -> list[list[float]] | None:
        """
        Get or set the load vectors on many faces.

        Parameters
        ----------
        load :
            The load vector to set on each face. If None, current loads are returned.
        keys :
            The faces to access. If None, all faces are used.

        Returns
        -------
        loads :
            The load vector of each face.
        """
        # names given as a non-empty tuple always returns a list of lists
        return self.faces_attributes(names=("px", "py", "pz"), values=load, keys=keys)  # pyright: ignore[reportReturnType]

    # ----------------------------------------------------------------------
    # Datastructure properties
    # ----------------------------------------------------------------------

    def parameters(self) -> tuple[list[float], list[list[float]], list[list[float]]]:
        """
        Return the force density design parameters of the mesh.

        Returns
        -------
        parameters :
            The edge force densities, the fixed vertex coordinates, and the vertex
            loads.
        """
        q = self.edges_forcedensities()
        xyz_fixed = self.vertices_fixedcoordinates()
        loads = self.vertices_loads()

        # getter-mode calls always return lists, never None
        assert q is not None
        assert loads is not None

        return q, xyz_fixed, loads

    # ----------------------------------------------------------------------
    # Maps
    # ----------------------------------------------------------------------

    def index_edge(self) -> dict[int, tuple[int, int]]:
        """
        Map each edge's enumeration index to its vertex key pair.

        Returns
        -------
        index_edge :
            A mapping from edge index to its ``(u, v)`` vertex key pair.

        Notes
        -----
        Mirrors ``compas.datastructures.Graph.index_edge``, which COMPAS 2.x does
        not provide on ``Mesh``, so ``FDMesh`` and ``FDNetwork`` share the API.
        """
        # data=False getter always yields (u, v) vertex-key pairs
        return dict(enumerate(self.edges()))

    def uv_index(self) -> dict[tuple[int, int], int]:
        """
        Map each edge's vertex key pair to its enumeration index.

        Returns
        -------
        uv_index :
            A mapping from each ``(u, v)`` vertex key pair to its edge index.
        """
        # data=False getter always yields (u, v) vertex-key pairs
        return {(u, v): index for index, (u, v) in enumerate(self.edges())}

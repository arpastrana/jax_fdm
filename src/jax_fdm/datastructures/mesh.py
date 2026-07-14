"""
A force density mesh.
"""
from collections.abc import Generator
from typing import Any

import numpy as np

from compas.datastructures import Mesh
from jax_fdm.datastructures import FDDatastructure
from jax_fdm.geometry import polygon_lcs


class FDMesh(Mesh, FDDatastructure):
    """
    A force density mesh.
    """
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        self.update_default_edge_attributes({"q": 0.0,
                                             "length": 0.0,
                                             "force": 0.0,
                                             "px": 0.0,
                                             "py": 0.0,
                                             "pz": 0.0})

        self.update_default_vertex_attributes({"x": 0.0,
                                               "y": 0.0,
                                               "z": 0.0,
                                               "px": 0.0,
                                               "py": 0.0,
                                               "pz": 0.0,
                                               "rx": 0.0,
                                               "ry": 0.0,
                                               "rz": 0.0,
                                               "is_support": False})

        self.update_default_face_attributes({"px": 0.0,
                                             "py": 0.0,
                                             "pz": 0.0})

    # ----------------------------------------------------------------------
    # Node
    # ----------------------------------------------------------------------

    def is_vertex_support(self, key: int) -> bool:
        """
        Test if the vertex is a support.
        """
        return self.vertex_attribute(key, name="is_support")  # pyright: ignore[reportReturnType]  # getter-mode call always returns the bool default

    def number_of_supports(self) -> int:
        """
        The number of supported vertices.
        """
        return len(list(self.vertices_supports()))  # pyright: ignore[reportArgumentType]  # vertices_supports() with no keys always returns a generator, never None

    def vertex_support(self, key: int) -> None:
        """
        Sets a vertex to a fixed support.
        """
        return self.vertex_attribute(key, name="is_support", value=True)  # pyright: ignore[reportReturnType]  # setter-mode call always returns None

    def vertex_load(self, key: int, load: list[float] | None = None) -> list[float] | None:
        """
        Gets or sets a load to a vertex.
        """
        return self.vertex_attributes(key, names=("px", "py", "pz"), values=load)  # pyright: ignore[reportReturnType]  # names given as a non-empty tuple always returns a list

    def vertex_residual(self, key: int) -> list[float]:
        """
        Gets the residual force of a mesh vertex.
        """
        return self.vertex_attributes(key, names=("rx", "ry", "rz"))  # pyright: ignore[reportReturnType]  # names given as a non-empty tuple always returns a list

    def vertex_reaction(self, key: int) -> list[float]:
        """
        Gets the reaction force of a mesh vertex.
        """
        return self.vertex_residual(key)

    def vertices_coordinates(self, keys: list[int] | None = None, axes: str = "xyz") -> list[list[float]]:
        """
        Gets or sets the x, y, z coordinates of a list of vertices.
        """
        vertex_keys = keys or self.vertices()
        return [self.vertex_coordinates(node, axes) for node in vertex_keys]  # pyright: ignore[reportReturnType]  # inherited COMPAS getter always returns a coordinate list here

    def vertices_fixedcoordinates(self, keys: list[int] | None = None, axes: str = "xyz") -> list[list[float]]:
        """
        Gets the x, y, z coordinates of the supports of the network.
        """
        if keys:
            vertex_keys = {key for key in keys if self.is_vertex_support(key)}
        else:
            vertex_keys = self.vertices_fixed()

        return [self.vertex_coordinates(node, axes) for node in vertex_keys]  # pyright: ignore[reportOptionalIterable,reportReturnType]  # vertices_fixed() with no keys always returns a generator, never None; getter always returns a coordinate list

    def vertices_supports(self, keys: list[int] | None = None) -> Generator[int, None, None] | None:
        """
        Gets or sets the vertex keys where a support has been assigned.
        """
        if keys is None:
            return self.vertices_where({"is_support": True})  # pyright: ignore[reportReturnType]  # data=False getter always yields plain vertex keys

        return self.vertices_attribute(name="is_support", value=True, keys=keys)  # pyright: ignore[reportReturnType]  # setter-mode call always returns None

    def vertices_fixed(self, keys: list[int] | None = None) -> Generator[int, None, None] | None:
        """
        Gets or sets the vertex keys where a support has been assigned.
        """
        return self.vertices_supports(keys)

    def vertices_free(self) -> Generator[int, None, None]:
        """
        The keys of the vertices where there is no support assigned.
        """
        return self.vertices_where({"is_support": False})  # pyright: ignore[reportReturnType]  # data=False getter always yields plain vertex keys

    def vertices_loads(self, load: list[float] | None = None, keys: list[int] | None = None) -> list[list[float]] | None:
        """
        Gets or sets a load to the vertices of the mesh.
        """
        return self.vertices_attributes(names=("px", "py", "pz"), values=load, keys=keys)  # pyright: ignore[reportReturnType]  # names given as a non-empty tuple always returns a list of lists

    def vertices_residual(self, keys: list[int] | None = None) -> list[list[float]]:
        """
        Gets the residual forces at the vertices of the mesh.
        """
        return self.vertices_attributes(names=("rx", "ry", "rz"), keys=keys)  # pyright: ignore[reportReturnType]  # names given as a non-empty tuple always returns a list of lists

    def vertices_reactions(self, keys: list[int] | None = None) -> list[list[float]]:
        """
        Gets the reaction forces at the vertices of the mesh.
        """
        keys = keys or self.vertices_fixed()  # pyright: ignore[reportAssignmentType]  # vertices_fixed() with no keys always returns a generator, never None
        return self.vertices_residual(keys)

    # ----------------------------------------------------------------------
    # Edges
    # ----------------------------------------------------------------------

    def is_edge_supported(self, key: tuple[int, int]) -> bool:
        """
        Test if any of edge vertices is a support.
        """
        return any(self.is_vertex_support(vertex) for vertex in key)

    def is_edge_fully_supported(self, key: tuple[int, int]) -> bool:
        """
        Test if all the edge vertices are a support.
        """
        return all(self.is_vertex_support(vertex) for vertex in key)

    # ----------------------------------------------------------------------
    # Faces
    # ----------------------------------------------------------------------

    def face_lcs(self, key: int) -> list[list[float]]:
        """
        Calculate the local coordinate system (LCS) of this face.
        """
        # np.asarray boundary into a jax-array-typed function; safe at runtime.
        return polygon_lcs(np.asarray(self.face_coordinates(key))).tolist()  # pyright: ignore[reportArgumentType]  # numpy/jax boundary

    def face_load(self, key: int, load: list[float] | None = None) -> list[float] | None:
        """
        Gets or sets a load on a face.
        """
        return self.face_attributes(key=key, names=("px", "py", "pz"), values=load)  # pyright: ignore[reportReturnType]  # names given as a non-empty tuple always returns a list

    def is_face_supported(self, key: int) -> bool:
        """
        Test if any of the face vertices is a support.
        """
        return any(self.is_vertex_support(vertex) for vertex in self.face_vertices(key))

    def is_face_fully_supported(self, key: int) -> bool:
        """
        Test if all the face vertices are a support.
        """
        return all(self.is_vertex_support(vertex) for vertex in self.face_vertices(key))

    def faces_loads(self, load: list[float] | None = None, keys: list[int] | None = None) -> list[list[float]] | None:
        """
        Gets or sets a load on the faces.
        """
        return self.faces_attributes(names=("px", "py", "pz"), values=load, keys=keys)  # pyright: ignore[reportReturnType]  # names given as a non-empty tuple always returns a list of lists

    # ----------------------------------------------------------------------
    # Datastructure properties
    # ----------------------------------------------------------------------

    def parameters(self) -> tuple[list[float] | None, list[list[float]], list[list[float]] | None]:
        """
        Return the design parameters of the network.
        """
        q = self.edges_forcedensities()
        xyz_fixed = self.vertices_fixedcoordinates()
        loads = self.vertices_loads()

        return q, xyz_fixed, loads

    # ----------------------------------------------------------------------
    # Maps
    # ----------------------------------------------------------------------

    def index_edge(self) -> dict[int, tuple[int, int]]:
        """
        Returns a dictionary that maps edges in a list to the corresponding vertex key pairs.

        Mirrors ``compas.datastructures.Graph.index_edge``, which compas 2.x does
        not provide on ``Mesh``, so ``FDMesh`` and ``FDNetwork`` share the API.
        """
        return dict(enumerate(self.edges()))  # pyright: ignore[reportReturnType]  # data=False getter always yields (u, v) vertex-key pairs

    def uv_index(self) -> dict[tuple[int, int], int]:
        """
        Returns a dictionary that maps edge keys (i.e. pairs of vertex keys)
        to the corresponding edge index in a list or array of edges.
        """
        return {(u, v): index for index, (u, v) in enumerate(self.edges())}  # pyright: ignore[reportReturnType]  # data=False getter always yields (u, v) vertex-key pairs

"""
A force density mesh.
"""
import numpy as np

from compas.datastructures import Mesh

from jax_fdm.datastructures import FDDatastructure

from jax_fdm.geometry import polygon_lcs


class FDMesh(Mesh, FDDatastructure):
    """
    A force density mesh.
    """
    def __init__(self, *args, **kwargs):
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

    def is_vertex_support(self, key):
        """
        Test if the vertex is a support.
        """
        return self.vertex_attribute(key, name="is_support")

    def number_of_supports(self):
        """
        The number of supported vertices.
        """
        return len(list(self.vertices_supports()))

    def vertex_support(self, key):
        """
        Sets a vertex to a fixed support.
        """
        return self.vertex_attribute(key, name="is_support", value=True)

    def vertex_load(self, key, load=None):
        """
        Gets or sets a load to a vertex.
        """
        return self.vertex_attributes(key, names=("px", "py", "pz"), values=load)

    def vertex_residual(self, key):
        """
        Gets the residual force of a mesh vertex.
        """
        return self.vertex_attributes(key, names=("rx", "ry", "rz"))

    def vertex_reaction(self, key):
        """
        Gets the reaction force of a mesh vertex.
        """
        return self.vertex_residual(key)

    def vertices_coordinates(self, keys=None, axes="xyz"):
        """
        Gets or sets the x, y, z coordinates of a list of vertices.
        """
        keys = keys or self.vertices()
        return [self.vertex_coordinates(node, axes) for node in keys]

    def vertices_fixedcoordinates(self, keys=None, axes="xyz"):
        """
        Gets the x, y, z coordinates of the supports of the network.
        """
        if keys:
            keys = {key for key in keys if self.is_vertex_support(key)}
        else:
            keys = self.vertices_fixed()

        return [self.vertex_coordinates(node, axes) for node in keys]

    def vertices_supports(self, keys=None):
        """
        Gets or sets the vertex keys where a support has been assigned.
        """
        if keys is None:
            return self.vertices_where({"is_support": True})

        return self.vertices_attribute(name="is_support", value=True, keys=keys)

    def vertices_fixed(self, keys=None):
        """
        Gets or sets the vertex keys where a support has been assigned.
        """
        return self.vertices_supports(keys)

    def vertices_free(self):
        """
        The keys of the vertices where there is no support assigned.
        """
        return self.vertices_where({"is_support": False})

    def vertices_loads(self, load=None, keys=None):
        """
        Gets or sets a load to the vertices of the mesh.
        """
        return self.vertices_attributes(names=("px", "py", "pz"), values=load, keys=keys)

    def vertices_residual(self, keys=None):
        """
        Gets the residual forces at the vertices of the mesh.
        """
        return self.vertices_attributes(names=("rx", "ry", "rz"), keys=keys)

    def vertices_reactions(self, keys=None):
        """
        Gets the reaction forces at the vertices of the mesh.
        """
        keys = keys or self.vertices_fixed()
        return self.vertices_residual(keys)

    # ----------------------------------------------------------------------
    # Edges
    # ----------------------------------------------------------------------

    def is_edge_supported(self, key):
        """
        Test if any of edge vertices is a support.
        """
        return any(self.is_vertex_support(vertex) for vertex in key)

    def is_edge_fully_supported(self, key):
        """
        Test if all the edge vertices are a support.
        """
        return all(self.is_vertex_support(vertex) for vertex in key)

    # ----------------------------------------------------------------------
    # Faces
    # ----------------------------------------------------------------------

    def face_lcs(self, key):
        """
        Calculate the local coordinate system (LCS) of this face.
        """
        return polygon_lcs(np.asarray(self.face_coordinates(key))).tolist()

    def face_load(self, key, load=None):
        """
        Gets or sets a load on a face.
        """
        return self.face_attributes(key=key, names=("px", "py", "pz"), values=load)

    def is_face_supported(self, key):
        """
        Test if any of the face vertices is a support.
        """
        return any(self.is_vertex_support(vertex) for vertex in self.face_vertices(key))

    def is_face_fully_supported(self, key):
        """
        Test if all the face vertices are a support.
        """
        return all(self.is_vertex_support(vertex) for vertex in self.face_vertices(key))

    def faces_loads(self, load=None, keys=None):
        """
        Gets or sets a load on the faces.
        """
        return self.faces_attributes(names=("px", "py", "pz"), values=load, keys=keys)

    # ----------------------------------------------------------------------
    # Datastructure properties
    # ----------------------------------------------------------------------

    def parameters(self):
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

    def index_uv(self):
        """
        Returns a dictionary that maps edges in a list to the corresponding vertex key pairs.
        """
        return dict(enumerate(self.edges()))

    def uv_index(self):
        """
        Returns a dictionary that maps edge keys (i.e. pairs of vertex keys)
        to the corresponding edge index in a list or array of edges.
        """
        return {(u, v): index for index, (u, v) in enumerate(self.edges())}

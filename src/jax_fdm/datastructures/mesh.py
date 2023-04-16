"""
A catalogue of force density datastructures.
"""

from math import fabs

from compas.datastructures import Mesh
from compas.geometry import transform_points

from jax_fdm.datastructures import NodeMixins


class FDMesh(Mesh, NodeMixins):
    """
    A force density mesh.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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

        self.update_default_edge_attributes({"q": 0.0,
                                             "length": 0.0,
                                             "force": 0.0,
                                             "px": 0.0,
                                             "py": 0.0,
                                             "pz": 0.0})

        self.update_default_face_attributes({"px": 0.0,
                                             "py": 0.0,
                                             "pz": 0.0})

    def vertices_coordinates(self, keys=None, axes="xyz"):
        """
        Gets or sets the x, y, z coordinates of a list of vertices.
        """
        keys = keys or self.vertices()

        return [self.vertex_coordinates(node, axes) for node in keys]

    def vertices_fixedcoordinates(self, keys=None, axes="xyz"):
        """
        Gets the x, y, z coordinates of the support vertices of the network.
        """
        if keys:
            keys = {key for key in keys if self.is_vertex_support(key)}
        else:
            keys = self.vertices_fixed()

        return [self.vertex_coordinates(node, axes) for node in keys]

    def number_of_anchors(self):
        """
        The number of anchored vertices.
        """
        return len(list(self.vertices_anchors()))

    def number_of_supports(self):
        """
        The number of supports.
        """
        return len(list(self.vertices_supports()))

    def vertex_support(self, key):
        """
        Sets a vertex as a support.
        """
        return self.vertex_attribute(key=key, name="is_support", value=True)

    def vertex_anchor(self, key):
        """
        Sets a vertex as a support.
        """
        return self.vertex_support(key)

    def is_vertex_support(self, key):
        """
        Test if the vertex is a support.
        """
        return self.vertex_attribute(key=key, name="is_support")

    def is_edge_supported(self, key):
        """
        Test if any of the two vertices connected by the edge is a vertex.
        """
        return any(self.is_vertex_support(node) for node in key)

    def is_edge_fully_supported(self, key):
        """
        Test if both vertices connected the edge is a vertex.
        """
        return all(self.is_vertex_support(node) for node in key)

    def vertices_supports(self, keys=None):
        """
        Gets or sets the vertices where a support is assigned.
        """
        if keys is None:
            return self.vertices_where({"is_support": True})

        return self.vertices_attribute(name="is_support", value=True, keys=keys)

    def vertices_fixed(self, keys=None):
        """
        Gets or sets the supported vertices.
        """
        return self.vertices_supports(keys)

    def vertices_anchors(self, keys=None):
        """
        Gets or sets the supported vertices.
        """
        return self.vertices_supports(keys)

    def vertices_free(self):
        """
        The unsupported vertices.
        """
        return self.vertices_where({"is_support": False})

    def edge_forcedensity(self, key, q=None):
        """
        Gets or sets the force density on a single edge.
        """
        return self.edge_attribute(name="q", value=q, edge=key)

    def edges_forcedensities(self, q=None, keys=None):
        """
        Gets or sets the force densities on a list of edges.
        """
        return self.edges_attribute(name="q", value=q, keys=keys)

    def vertex_load(self, key, load=None):
        """
        Gets or sets the load on a vertex.
        """
        return self.vertex_attributes(key=key, names=("px", "py", "pz"), values=load)

    def vertices_loads(self, load=None, keys=None):
        """
        Gets or sets the load on the vertices.
        """
        return self.vertices_attributes(names=("px", "py", "pz"), values=load, keys=keys)

    def edge_load(self, key, load=None):
        """
        Gets or sets the load on a vertex.
        """
        return self.edge_attributes(edge=key, names=("px", "py", "pz"), values=load)

    def edges_loads(self, load=None, keys=None):
        """
        Gets or sets a load on the edges.
        """
        return self.edges_attributes(names=("px", "py", "pz"), values=load, keys=keys)

    def face_load(self, key, load=None):
        """
        Gets or sets the load on a face.
        """
        return self.face_attributes(key=key, names=("px", "py", "pz"), values=load)

    def faces_loads(self, load=None, keys=None):
        """
        Gets or sets a load on the faces.
        """
        return self.faces_attributes(names=("px", "py", "pz"), values=load, keys=keys)

    def vertices_residual(self, keys=None):
        """
        Gets the reaction forces at the vertices.
        """
        return self.vertices_attributes(names=("rx", "ry", "rz"), keys=keys)

    def vertex_residual(self, key):
        """
        Gets the reaction force of a vertex.
        """
        return self.vertex_attributes(key=key, names=("rx", "ry", "rz"))

    def vertices_reactions(self, keys=None):
        """
        Gets the reaction forces at the vertices.
        """
        keys = keys or self.vertices_fixed()

        return self.vertices_attributes(names=("rx", "ry", "rz"), keys=keys)

    def vertex_reaction(self, key):
        """
        Gets the reaction force at a vertex.
        """
        return self.vertex_attributes(key=key, names=("rx", "ry", "rz"))

    def edge_force(self, key):
        """
        Gets the internal force at an edge.
        """
        return self.edge_attribute(edge=key, name="force")

    def edges_forces(self, keys=None):
        """
        Gets the edges internal force.
        """
        return self.edges_attribute(keys=keys, name="force")

    def edges_lengths(self, keys=None):
        """
        Gets the edge lengths.
        """
        return self.edges_attribute(keys=keys, name="length")

    def edge_loadpath(self, key):
        """
        Gets the load path of an edge.
        """
        force = self.edge_attribute(edge=key, name="force")
        length = self.edge_attribute(edge=key, name="length")

        return fabs(force * length)

    def edges_loadpaths(self, keys=None):
        """
        Yields the load path of all the edges.
        """
        keys = keys or self.edges()

        for key in keys:
            yield self.edge_loadpath(key)

    def loadpath(self):
        """
        Gets the total load path of the mesh.
        """
        return sum(list(self.edges_loadpaths()))

    def parameters(self):
        """
        Return the design parameters of the mesh.
        """
        q = self.edges_forcedensities()
        xyz_fixed = self.vertices_fixedcoordinates()
        # loads = self.faces_loads()
        loads = self.vertices_loads()

        return q, xyz_fixed, loads

    def print_stats(self, other_stats=None):
        """
        Print information aboud the equilibrium state of the mesh.
        """
        stats = {"FDs": self.edges_forcedensities(),
                 "Forces": self.edges_forces(),
                 "Lengths": self.edges_lengths()}

        other_stats = other_stats or {}
        stats.update(other_stats)

        print("\n***Mesh stats***")
        print(f"Load path: {round(self.loadpath(), 3)}")

        for name, vals in stats.items():

            minv = round(min(vals), 3)
            maxv = round(max(vals), 3)
            meanv = round(sum(vals) / len(vals), 3)

            print(f"{name}\t\tMin: {minv}\tMax: {maxv}\tMean: {meanv}")

    def transformed(self, transformation):
        """
        Return a transformed copy of the network.
        """
        mesh = self.copy()
        vertices = list(self.vertices())

        attr_groups = [("x", "y", "z")]

        for attr_names in attr_groups:
            attrs = self.vertices_attributes(names=attr_names, keys=nodes)
            attrs_t = transform_points(attrs, transformation)

            for vertex, values in zip(vertices, attrs_t):
                mesh.vertex_attributes(vertex, names=attr_names, values=values)

        return mesh

    def index_uv(self):
        """
        Returns a dictionary that maps edges in a list to the corresponding vertex key pairs.
        """
        return dict(enumerate(self.edges()))


if __name__ == "__main__":

    import os
    from jax import jit

    from compas.datastructures import Mesh

    from jax_fdm import DATA
    from jax_fdm.datastructures import FDMesh
    from jax_fdm.equilibrium import EquilibriumStructureMesh
    from jax_fdm.equilibrium import fdm

    from jax_fdm.visualization import Viewer

    mesh = FDMesh.from_meshgrid(dx=5, nx=5)

    mesh.vertices_supports(mesh.vertices_on_boundary())
    mesh.edges_forcedensities(-1.0)
    # mesh.faces_loads([0.0, 0.0, -1.0])
    mesh.vertices_loads([0.0, 0.0, -1.0])

    mesh_eq = fdm(mesh)

    viewer = Viewer()
    viewer.add(mesh_eq)
    viewer.show()

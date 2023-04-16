import numpy as np
import jax.numpy as jnp

from jax import vmap

from jax_fdm.geometry import area_triangle
from jax_fdm.geometry import length_vector

# ==========================================================================
# Loads sourcery
# ==========================================================================

# LoadFaces
# LoadEdges
# LoadNodes

class LoadCalculator:
    """
    An object that calculates loads applied to the nodes, edges and faces of a structure.
    """
    def __init__(self, structure):
        """
        Initialize the calculator.
        """
        self.face_calculator = LoadFaces(structure)
        mesh = structure.datastruct
        self.face_loads = jnp.asarray([mesh.face_load(fkey) for fkey in mesh.faces()])

    def __call__(self, xyz, loads):
        """
        Calculate the current loads applied to the nodes of the structure.
        """
        # load_nodes = self.load_nodes(xyz)
        # load_edges = self.load_edges(xyz)
        # load_faces = self.load_faces(xyz)

        # return loads + load_nodes + load_edges + load_faces
        return loads + self.face_calculator(xyz, self.face_loads)

# ==========================================================================
# Loads
# ==========================================================================

    def load_faces(self, xyz, face_loads):
        """
        Update face area loads based on the current XYZ coordinates of a structure.
        """

        # compute edge tributary areas from centroids
        # face_loads = self.model.loads.faces  # TODO: to implement
        edge_loads = self.edges_tributary_load(xyz, face_loads)

        # compute node tributary load
        node_loads = self.nodes_tributary_load(edge_loads)

        return node_loads

    def load_nodes(self, xyz):
        """
        Update node point loads based on the current XYZ coordinates of a structure.

        TODO: To be implemented later for loads that change direction with XYZ.
        """
        return 0.0

    def load_edges(self, xyz):
        """
        Update edge line loads based on the current XYZ coordinates of a structure.

        # To be implemented later.
        """
        return 0.0

# ==========================================================================
# Face loads
# ==========================================================================

class LoadFaces:
    """
    Manage and compute area loads applied to the faces of a structure.
    """
    def __init__(self, structure):
        self.structure = structure

    def __call__(self, xyz, faces_load):
        """
        Update face area loads based on the current XYZ coordinates of a structure.
        """
        # face_loads = self.model.loads.faces  # TODO: to implement
        # compute edge tributary areas from centroids
        edges_load = self.edges_tributary_load(xyz, faces_load)

        # compute node tributary load
        nodes_load = self.nodes_tributary_load(edges_load)

        return nodes_load

    def nodes_tributary_load(self, edges_load):
        """
        Calculate the load vector applied to the nodes based on the edge loads.
        """
        return 0.5 * jnp.abs(self.structure.connectivity).T @ edges_load

    def edges_tributary_load(self, xyz, faces_load):
        """
        Calculate the face area load taken by every edge in a datastructure.
        """
        face_centroids = self.structure.connectivity_faces @ xyz
        edges = jnp.asarray(self.structure.edges)
        connectivity = self.structure.connectivity_edges_faces
        loads_fn = vmap(self.edge_tributary_load, in_axes=(0, 0, None, None, None))

        return loads_fn(edges, connectivity, xyz, faces_load, face_centroids)

    def edge_tributary_load(self, edge, connectivity_edge, xyz, faces_load, face_centroids):
        """
        Calculate the face area load taken by one edge in a datastructure.
        """
        def tributary_area(line, centroid):
            triangle = jnp.vstack((line, jnp.reshape(centroid, (1, 3))))
            return area_triangle(triangle)

        findices = jnp.flatnonzero(connectivity_edge, size=2, fill_value=-1)
        line = xyz[edge, :]
        centroids = face_centroids[findices, :]

        # correct loads for negative face indices
        areas = vmap(tributary_area, in_axes=(None, 0))(line, centroids)
        areas = jnp.where(findices >= 0, areas.ravel(), 0.0)

        floads = faces_load[findices, :]

        return areas @ floads


# ==========================================================================
# Edge loads
# ==========================================================================

class LoadEdges:
    """
    Manage and compute area loads applied to the edges of a structure.
    """
    def __init__(self, structure):
        self.structure = structure

    def __call__(self, xyz, edges_load):
        """
        Update face area loads based on the current XYZ coordinates of a structure.
        """
        # face_loads = self.model.loads.faces  # TODO: to implement
        # compute edge tributary areas from centroids
        edges_load = self.edges_tributary_load(xyz, edges_load)

        # compute node tributary load
        nodes_load = self.nodes_tributary_load(edges_load)

        return nodes_load

    def nodes_tributary_load(self, edges_load):
        """
        Calculate the load vector applied to the nodes based on the edge loads.
        """
        return 0.5 * jnp.abs(self.structure.connectivity).T @ edges_load

    def edges_tributary_load(self, xyz, edges_load):
        """
        Calculate the face area load taken by every edge in a datastructure.
        """
        # calculate edge lengths
        edges_vector = self.structure.connectivity @ xyz
        edges_length = length_vector(edges_vector)

        # scale edge load by edge length
        edges_load = edges_load * edges_length

        return edges_load


if __name__ == "__main__":
    import os
    from jax import jit

    from compas.datastructures import Mesh

    from jax_fdm import DATA
    from jax_fdm.datastructures import FDMesh
    from jax_fdm.equilibrium import EquilibriumStructureMesh


    def test_faces_load(mesh, pz, verbose=False):

        structure = EquilibriumStructureMesh(mesh)
        calculator = LoadFaces(structure)

        xyz = jnp.array([mesh.vertex_coordinates(vkey) for vkey in mesh.vertices()])
        faces_load = jnp.zeros((mesh.number_of_faces(), 3))
        faces_load = faces_load.at[:, 2].set(pz)
        edges_load = calculator.edges_tributary_load(xyz, faces_load)
        nodes_load = calculator(xyz, faces_load)

        sum_nodes_load = jnp.sum(jnp.stack(nodes_load))
        sum_edges_load = jnp.sum(jnp.stack(edges_load))

        compas_load = mesh.area() * pz

        print(f"{sum_nodes_load=:.3f}, {sum_edges_load=:.3f}, {compas_load=:.3f}")

        assert jnp.allclose(sum_edges_load, compas_load)
        assert jnp.allclose(sum_nodes_load, compas_load)

        if verbose:
            print("num edges", mesh.number_of_edges())
            print("num nodes", mesh.number_of_vertices())
            print("edges load shape", edges_load.shape)
            print("nodes load shape", nodes_load.shape)
            print("faces load shape", faces_load.shape)


    def test_edges_load(mesh, pz, verbose=False):

        structure = EquilibriumStructureMesh(mesh)
        calculator = LoadEdges(structure)

        xyz = jnp.array([mesh.vertex_coordinates(vkey) for vkey in mesh.vertices()])
        edges_load = jnp.zeros((mesh.number_of_edges(), 3))
        edges_load = edges_load.at[:, 2].set(pz)
        nodes_load = calculator(xyz, edges_load)
        edges_load = calculator.edges_tributary_load(xyz, edges_load)
        sum_nodes_load = jnp.sum(jnp.stack(nodes_load))
        sum_edges_load = jnp.sum(jnp.stack(edges_load))

        compas_load = sum(mesh.edge_length(u, v) * pz for u, v in mesh.edges())

        print(f"{sum_nodes_load=:.3f}, {sum_edges_load=:.3f}, {compas_load=:.3f}")

        if verbose:
            print("num edges", mesh.number_of_edges())
            print("num nodes", mesh.number_of_vertices())
            print("edges load shape", edges_load.shape)
            print("nodes load shape", nodes_load.shape)

        assert jnp.allclose(sum_edges_load, compas_load)
        assert jnp.allclose(sum_nodes_load, compas_load)

    pz = 1.0

    for nx in range(5, 10):
        mesh = FDMesh.from_meshgrid(dx=2, nx=nx)
        test_faces_load(mesh, pz)
        test_edges_load(mesh, pz)
        print()

    for f in (4, 6, 8, 12, 20):
        mesh = FDMesh.from_polyhedron(f)
        test_faces_load(mesh, pz)
        test_edges_load(mesh, pz)
        print()

    for name in ("uncstr_opt", "cstr_opt", "free", "loaded"):
        filepath = os.path.join(DATA, f"json/pillow_{name}_mesh.json")
        mesh = FDMesh.from_json(filepath)
        test_faces_load(mesh, pz)
        test_edges_load(mesh, pz)
        print()

    print("\nVery good! Slow down, cowboy!")

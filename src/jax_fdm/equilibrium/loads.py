import numpy as np
import jax.numpy as jnp

from jax import vmap

from jax_fdm.geometry import area_triangle

# ==========================================================================
# Loads sourcery
# ==========================================================================

class LoadCalculator:
    """
    An object that calculates loads applied to the nodes, edges and faces of a structure.
    """
    def __init__(self, structure):
        """
        Initialize the calculator.
        """
        self.structure = structure

    def __call__(self, xyz, loads):
        """
        Calculate the current loads applied to the nodes of the structure.
        """
        load_nodes = self.load_nodes(xyz_all)
        load_edges = self.load_edges(xyz_all)
        load_faces = self.load_faces(xyz_all)

        return loads + load_nodes + load_edges + load_faces

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
# Tributes
# ==========================================================================

    def nodes_tributary_load(self, edge_loads):
        """
        Calculate the load vector applied to the nodes based on the edge loads.
        """
        return  0.5 * jnp.abs(self.structure.connectivity).T @ edge_loads

    def edges_tributary_load(self, xyz, faces_load):
        """
        Calculate the face area load taken by every edge in a datastructure.
        """
        # calculate face centroids
        face_centroids = self.structure.connectivity_faces @ xyz
        edges = jnp.asarray(self.structure.edges)
        connectivity = self.structure.connectivity_edges_faces
        loads_fn = vmap(self.edge_tributary_load, in_axes=(0, 0, None, None, None))

        return loads_fn(edges, connectivity, xyz, faces_load, face_centroids)

    def edge_tributary_load(self, edge, connectivity_edge, xyz, faces_load, face_centroids):
        """
        Calculate the face area load vector taken by one edge in a datastructure.
        """
        def tributary_area(line, centroid):
            triangle = jnp.vstack((line, jnp.reshape(centroid, (1, 3))))
            return area_triangle(triangle)

        findices = jnp.flatnonzero(connectivity_edge, size=2, fill_value=-1)
        line = xyz[edge, :]
        centroids = face_centroids[findices, :]
        areas = vmap(tributary_area, in_axes=(None, 0), out_axes=0)(line, centroids)
        # correct loads for negative face indices
        areas = jnp.where(findices >= 0, areas.ravel(), 0.0)

        floads = faces_load[findices, :]

        return areas @ floads


if __name__ == "__main__":
    from jax import jit

    from compas.datastructures import Mesh
    from jax_fdm.datastructures import FDMesh
    from jax_fdm.equilibrium import EquilibriumStructureMesh

    # print(FDMesh.__mro__)
    mesh = Mesh.from_meshgrid(dx=1, nx=50)
    mesh = FDMesh.from_meshgrid(dx=1, nx=50)  # TODO: refactor FDMesh

    structure = EquilibriumStructureMesh(mesh)
    calculator = LoadCalculator(structure)

    xyz = jnp.array([mesh.vertex_coordinates(vkey) for vkey in mesh.vertices()])
    faces_load = jnp.zeros((mesh.number_of_faces(), 3))
    faces_load = faces_load.at[:, 2].set(1.0)
    edges_load = calculator.edges_tributary_load(xyz, faces_load)

    print("num edges", mesh.number_of_edges())
    print("faces load shape", faces_load.shape)
    print("edges load shape", edges_load.shape)

    # print(edges_load)
    print(jnp.sum(jnp.stack(edges_load)), mesh.area())

    nodes_load = calculator.nodes_tributary_load(edges_load)
    print("nodes load shape", nodes_load.shape)
    print(jnp.sum(jnp.stack(nodes_load)), mesh.area())


    print("Very good! Slow down, cowboy!")

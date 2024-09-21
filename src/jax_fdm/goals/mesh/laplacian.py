import jax.numpy as jnp

from jax import vmap

from jax_fdm.goals import ScalarGoal

from jax_fdm.goals.mesh import MeshGoal
from jax_fdm.goals.network import NetworkXYZLaplacianGoal


class MeshXYZLaplacianGoal(NetworkXYZLaplacianGoal):
    """
    A thin wrapper of NetworkXYZLaplacianGoal.
    """
    def __init__(self, target=0.0, weight=1.0):
        super().__init__(key=-1, target=target, weight=weight)


class MeshXYZFaceLaplacianGoal(ScalarGoal, MeshGoal):
    """
    Minimize the Laplacian energy of the XYZ faces coordinates of a mesh.

    Notes
    -----
    This goal can be handy to "smoothen" the looks of a mesh.

    This energy is computed as the distance of the XYZ coordinates of
    every vertex to the centroid of the centroids of the faces the
    vertex is part of.

    An enegy-minimizing mesh will have every vertex as close as possible
    to its neighboring faces centroid.
    """
    def __init__(self, target=0.0, weight=1.0):
        super().__init__(key=-1, target=target, weight=weight)
        self.connectivity_faces_vertices = None

    def init(self, model, structure):
        """
        Initialize the goal with information of a model and a structure.
        """
        super().init(model, structure)
        self.connectivity_faces_vertices = structure.connectivity_faces_vertices

    def laplacian_vertices(self, eq_state, index):
        """
        The current mesh Laplacian.
        """
        def vertex_laplacian(vertex_xyz, vertex_mask, faces_centroid):
            nbrs_xyz = vertex_faces_centroid(vertex_mask, faces_centroid)
            return vertex_square_distance(vertex_xyz, nbrs_xyz)

        def vertex_faces_centroid(vertex_mask, faces_centroid):
            nbrs_xyzs = jnp.reshape(vertex_mask, (-1, 1)) * faces_centroid
            # return average centroid
            return jnp.sum(nbrs_xyzs, axis=0) / jnp.sum(vertex_mask)

        def vertex_square_distance(vertex_xyz, nbrs_xyz):
            return jnp.sum(jnp.square(vertex_xyz - nbrs_xyz))

        faces_centroid = self.connectivity_faces_vertices @ eq_state.xyz
        vertices_laplacian = vmap(vertex_laplacian, in_axes=(0, 1, None))
        laplacians = vertices_laplacian(eq_state.xyz,
                                        self.connectivity_faces_vertices,
                                        faces_centroid)

        return laplacians

    def prediction(self, eq_state, index):
        """
        The current mesh Laplacian.
        """
        laplacians = self.laplacian_vertices(eq_state, index)

        return jnp.atleast_1d(jnp.mean(laplacians))

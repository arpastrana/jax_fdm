import jax.numpy as jnp
from jax import vmap
from jaxtyping import Array
from jaxtyping import Float
from jaxtyping import Int

from jax_fdm.equilibrium import EquilibriumMeshStructure
from jax_fdm.equilibrium import EquilibriumModel
from jax_fdm.equilibrium import EquilibriumState
from jax_fdm.goals import ScalarGoal
from jax_fdm.goals.mesh import MeshGoal
from jax_fdm.goals.network import NetworkXYZLaplacianGoal


class MeshXYZLaplacianGoal(NetworkXYZLaplacianGoal):
    """
    Minimize the Laplacian energy of a mesh's coordinates.

    Notes
    -----
    A thin wrapper of
    [NetworkXYZLaplacianGoal][jax_fdm.goals.network.laplacian.NetworkXYZLaplacianGoal]
    that fixes the key to the mesh sentinel.
    """

    def __init__(
        self,
        target: float | Float[Array, "..."] = 0.0,
        weight: float = 1.0,
    ) -> None:
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

    An energy-minimizing mesh will have every vertex as close as possible
    to its neighboring faces centroid.
    """

    def __init__(
        self,
        target: float | Float[Array, "..."] = 0.0,
        weight: float = 1.0,
    ) -> None:
        super().__init__(key=-1, target=target, weight=weight)
        # set in init() from the mesh structure, before any prediction runs
        self.connectivity_faces_vertices: Float[Array, "faces vertices"]

    def init(
        self,
        model: EquilibriumModel,
        structure: EquilibriumMeshStructure,
    ) -> None:
        """
        Bind the goal to a mesh, caching its face-vertex connectivity.

        Parameters
        ----------
        model :
            The equilibrium model.
        structure :
            The mesh structure whose face-vertex connectivity is cached.
        """
        super().init(model, structure)
        self.connectivity_faces_vertices = structure.connectivity_faces_vertices

    def laplacian_vertices(
        self,
        eq_state: EquilibriumState,
        index: Int[Array, "1"],
    ) -> Float[Array, "vertices"]:
        """
        The per-vertex Laplacian energy against neighboring face centroids.

        Parameters
        ----------
        eq_state :
            The equilibrium state to read vertex coordinates from.
        index :
            The sentinel index, unused.

        Returns
        -------
        laplacians :
            The squared distance from each vertex to the mean of its incident face
            centroids.
        """

        def vertex_laplacian(
            vertex_xyz: Float[Array, "3"],
            vertex_mask: Float[Array, "faces"],
            faces_centroid: Float[Array, "faces 3"],
        ) -> Float[Array, ""]:
            nbrs_xyz = vertex_faces_centroid(vertex_mask, faces_centroid)
            return vertex_square_distance(vertex_xyz, nbrs_xyz)

        def vertex_faces_centroid(
            vertex_mask: Float[Array, "faces"],
            faces_centroid: Float[Array, "faces 3"],
        ) -> Float[Array, "3"]:
            nbrs_xyzs = jnp.reshape(vertex_mask, (-1, 1)) * faces_centroid
            # return average centroid
            return jnp.sum(nbrs_xyzs, axis=0) / jnp.sum(vertex_mask)

        def vertex_square_distance(
            vertex_xyz: Float[Array, "3"],
            nbrs_xyz: Float[Array, "3"],
        ) -> Float[Array, ""]:
            return jnp.sum(jnp.square(vertex_xyz - nbrs_xyz))

        faces_centroid = self.connectivity_faces_vertices @ eq_state.xyz
        vertices_laplacian = vmap(vertex_laplacian, in_axes=(0, 1, None))
        laplacians = vertices_laplacian(
            eq_state.xyz,
            self.connectivity_faces_vertices,
            faces_centroid,
        )

        return laplacians

    def prediction(
        self,
        eq_state: EquilibriumState,
        index: Int[Array, "1"],
    ) -> Float[Array, ""]:
        """
        The mean per-vertex Laplacian energy of the mesh.

        Parameters
        ----------
        eq_state :
            The equilibrium state to read vertex coordinates from.
        index :
            The sentinel index, unused.

        Returns
        -------
        prediction :
            The mean over vertices of the face-centroid Laplacian energy.
        """
        laplacians = self.laplacian_vertices(eq_state, index)

        return jnp.mean(laplacians)

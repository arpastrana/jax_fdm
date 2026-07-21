import jax.numpy as jnp
from jax.experimental.sparse import BCOO
from jaxtyping import Array
from jaxtyping import Float
from jaxtyping import Int

from jax_fdm.equilibrium import EquilibriumMeshStructure
from jax_fdm.equilibrium import EquilibriumModel
from jax_fdm.equilibrium import EquilibriumState
from jax_fdm.goals import ScalarGoal
from jax_fdm.goals.goal import TargetLike
from jax_fdm.goals.mesh import MeshGoal
from jax_fdm.goals.network import NetworkXYZLaplacianGoal

__all__ = ["MeshXYZLaplacianGoal", "MeshXYZFaceLaplacianGoal"]


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
        target: TargetLike = 0.0,
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
        target: TargetLike = 0.0,
        weight: float = 1.0,
    ) -> None:
        super().__init__(key=-1, target=target, weight=weight)
        # set in init() from the mesh structure, before any prediction runs
        self.connectivity_faces_vertices: (
            Float[Array, "faces vertices"] | Float[BCOO, "faces vertices"]
        )

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

        Notes
        -----
        Formulated as products with the face-vertex connectivity so that the
        matrix can be stored dense or in sparse format. Each vertex's neighbor
        centroid is the average of its incident face centroids, weighted by the
        row-normalized connectivity entries.
        """
        xyz = eq_state.xyz
        connectivity = self.connectivity_faces_vertices

        faces_centroid = connectivity @ xyz

        # upstream types the sparse transpose as optional; it is None only for
        # arrays of dimension > 2, unreachable for a rank-2 incidence matrix
        connectivity_t = connectivity.T  # pyright: ignore[reportOptionalOperand]
        weights = connectivity_t @ jnp.ones(connectivity.shape[0])  # pyright: ignore[reportOptionalOperand]
        nbrs_centroid = (connectivity_t @ faces_centroid) / weights[:, None]  # pyright: ignore[reportOptionalOperand]

        return jnp.sum(jnp.square(xyz - nbrs_centroid), axis=-1)

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

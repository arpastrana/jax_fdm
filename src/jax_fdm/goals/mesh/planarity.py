import jax.numpy as jnp
from jax import vmap
from jaxtyping import Array
from jaxtyping import Float
from jaxtyping import Int

from jax_fdm.equilibrium import EquilibriumMeshStructure
from jax_fdm.equilibrium import EquilibriumModel
from jax_fdm.equilibrium import EquilibriumState
from jax_fdm.geometry import planarity_polygon
from jax_fdm.geometry import planarity_triangle
from jax_fdm.goals import ScalarGoal
from jax_fdm.goals.goal import TargetLike
from jax_fdm.goals.mesh import MeshGoal

__all__ = ["MeshPlanarityGoal", "face_xyz", "face_planarity", "faces_planarity"]


class MeshPlanarityGoal(ScalarGoal, MeshGoal):
    """
    Planarize the faces of a mesh.

    Notes
    -----
    This goal computes the average planarity of the faces of a mesh.
    The planarity of a face is the mean absolute dot product between the
    face's unitized normal vector and its unitized edge vectors, so faces of
    different degrees score comparably and ngons do not dominate the average.
    Triangles are planar by construction and contribute zero to the average;
    padded faces and faces with more than 4 vertices are handled correctly.

    The prediction is an energy, not a distance to a target. To minimize it
    as-is, pair this goal with a
    [PredictionError][jax_fdm.losses.errors.PredictionError]; a
    [SquaredError][jax_fdm.losses.errors.SquaredError] would square the
    energy, flattening its gradient as the faces approach planarity.
    """

    def __init__(
        self,
        target: TargetLike = 0.0,
        weight: float = 1.0,
    ) -> None:
        super().__init__(key=-1, target=target, weight=weight)
        # set in init() from the mesh structure, before any prediction runs
        self.faces_indexed: Int[Array, "faces vertices"]

    def init(
        self,
        model: EquilibriumModel,
        structure: EquilibriumMeshStructure,
    ) -> None:
        """
        Bind the goal to a mesh, caching its face indices.

        Parameters
        ----------
        model :
            The equilibrium model.
        structure :
            The mesh structure whose face indices are cached.
        """
        super().init(model, structure)
        self.faces_indexed = structure.faces_indexed

    def prediction(
        self,
        eq_state: EquilibriumState,
        index: Int[Array, "1"],
    ) -> Float[Array, ""]:
        """
        The average planarity of the mesh faces.

        Parameters
        ----------
        eq_state :
            The equilibrium state to read vertex coordinates from.
        index :
            The sentinel index, unused.

        Returns
        -------
        prediction :
            The mean face planarity, zero when every face is planar.
        """
        planarities = faces_planarity(self.faces_indexed, eq_state.xyz)

        return jnp.mean(planarities)


# ==========================================================================
# Planarity
# ==========================================================================


def face_xyz(
    face: Int[Array, "vertices"],
    xyz: Float[Array, "nodes 3"],
) -> Float[Array, "vertices 3"]:
    """
    Gather the coordinates of a face's vertices, padding safely for gradients.

    Parameters
    ----------
    face :
        The vertex indices of the face, with ``-1`` padding for absent vertices.
    xyz :
        The coordinates of all vertices.

    Returns
    -------
    xyz_face :
        The coordinates of the face's vertices.

    Notes
    -----
    Padding indices (``-1``) are replaced by the first vertex, so the rectangular
    face array indexes valid rows without producing nan gradients.
    """
    face = jnp.ravel(face)
    xyz_face = xyz[face, :]
    xyz_repl = xyz_face[0, :]

    return vmap(jnp.where, in_axes=(0, 0, None))(face >= 0, xyz_face, xyz_repl)


def face_planarity(
    face: Int[Array, "vertices"],
    xyz: Float[Array, "nodes 3"],
) -> Float[Array, ""]:
    """
    Compute a face's planarity, dispatching on its vertex count.

    Parameters
    ----------
    face :
        The vertex indices of the face.
    xyz :
        The coordinates of all vertices.

    Returns
    -------
    planarity :
        The mean absolute normal-edge cosine of the face, zero for a triangle
        by construction.

    Notes
    -----
    Faces with more than three valid vertices, including padded ones, take the
    polygon path; the padding scheme in `face_xyz` makes the extra rows inert.

    The polygon planarity is divided by the face's edge count, which for a
    closed polygon equals its valid vertex count and which the zero-cosine pad
    edges inflate neither. The per-edge mean keeps faces of different degrees
    comparable, so ngons do not dominate a mean over a mixed-degree mesh.
    """
    valid_face_indices = jnp.where(face >= 0, 1.0, 0.0)
    sum_indices = jnp.sum(valid_face_indices)

    fxyz = face_xyz(face, xyz)
    planarity = jnp.where(
        sum_indices == 3,
        planarity_triangle(fxyz),
        planarity_polygon(fxyz) / sum_indices,
    )

    return planarity


def faces_planarity(
    faces: Int[Array, "faces vertices"],
    xyz: Float[Array, "nodes 3"],
) -> Float[Array, "faces"]:
    """
    Compute the planarity of every face in a mesh.

    Parameters
    ----------
    faces :
        The vertex indices of each face.
    xyz :
        The coordinates of all vertices.

    Returns
    -------
    planarities :
        The planarity of each face.
    """
    return vmap(face_planarity, in_axes=(0, None))(faces, xyz)

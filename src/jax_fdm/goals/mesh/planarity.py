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
from jax_fdm.goals.mesh import MeshGoal


class MeshPlanarityGoal(ScalarGoal, MeshGoal):
    """
    Planarize the faces of a mesh.

    Notes
    -----
    This goal computes the average planarity of the faces of a mesh.
    The planarity of a face is calculated as the absolute dot product between
    the face's unitized normal vector and its unitized edge vectors.

    This function is experimental and it is unclear whether it works correctly
    for padded faces or faces with more than 4 vertices. Use with caution!
    """

    def __init__(
        self,
        target: float | Float[Array, "..."] = 0.0,
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
        index: Int[Array, ""],
    ) -> Float[Array, "1"]:
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

        return jnp.atleast_1d(jnp.mean(planarities))


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
        The face planarity, zero for a triangle by construction.
    """
    valid_face_indices = jnp.where(face >= 0, 1.0, 0.0)
    sum_indices = jnp.sum(valid_face_indices)

    fxyz = face_xyz(face, xyz)
    planarity = jnp.where(
        sum_indices == 3,
        planarity_triangle(fxyz),
        planarity_polygon(fxyz),
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

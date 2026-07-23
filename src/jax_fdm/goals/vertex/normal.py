from collections.abc import Sequence

import equinox as eqx
import jax.numpy as jnp
from jax import vmap
from jaxtyping import Array
from jaxtyping import Float
from jaxtyping import Int

from jax_fdm.equilibrium import EquilibriumMeshStructure
from jax_fdm.equilibrium import EquilibriumState
from jax_fdm.geometry import angle_vectors
from jax_fdm.geometry import normal_polygon
from jax_fdm.geometry import normalize_vector
from jax_fdm.goals.goal import TargetLike
from jax_fdm.goals.vertex.vertex import VertexGoal

__all__ = ["VertexNormalAngleGoal"]


def _as_vector(vector: Float[Array, "..."] | Sequence[float]) -> Float[Array, "3"]:
    """
    Coerce a reference vector to a JAX array.

    Parameters
    ----------
    vector :
        The reference vector; a flat xyz sequence or array.

    Returns
    -------
    vector :
        The reference vector as a JAX array, unbatched like the goal's other
        leaves.
    """
    return jnp.asarray(vector)


class VertexNormalAngleGoal(VertexGoal):
    """
    Drive the angle between a vertex normal and a reference vector toward a target.

    Parameters
    ----------
    key :
        The key of the vertex the goal acts on.
    target :
        The target angle, in radians.
    weight :
        The relative importance of the goal in the loss.
    vector :
        The reference vector each vertex normal's angle is measured against.
        Keyword-only and required, since there is no meaningful default
        reference direction.

    Notes
    -----
    The vertex normal is the unitized average of the normals of the faces
    surrounding the vertex, so this goal only applies to meshes.

    The angle is the arccosine of the signed cosine, so it spans [0, pi] and
    is covariant with the orientation of the vertex normal: a normal within
    90 degrees of the reference vector reads as acute, one folded past it as
    obtuse. Like the edge orientation in `EdgeAngleGoal`, the winding of the
    mesh faces is treated as data because it sets the normal's orientation.
    The mesh must therefore have a unified face winding for the averaged
    vertex normal, and hence the signed angle, to be meaningful. No runtime
    check is performed; `compas.datastructures.Mesh.unify_cycles` unifies
    the winding of a mesh upfront if needed.
    """

    # kw_only lets this required leaf follow the base's defaulted weight (the
    # dataclass "non-default after default" rule) without a default of its own.
    # A normal init field (not init=False), so equinox does not warn about a
    # differentiable leaf; the custom __init__ below sets and coerces it. Stored
    # unbatched like the goal's other leaves, so it is a plain (3,) vector this
    # element's prediction reads directly.
    vector: Float[Array, "3"] = eqx.field(kw_only=True)

    def __init__(
        self,
        key: int,
        target: TargetLike,
        weight: float = 1.0,
        *,
        vector: Float[Array, "..."] | Sequence[float],
    ) -> None:
        self.key = key
        self.target = target
        self.weight = weight
        self.vector = _as_vector(vector)

    @staticmethod
    def face_normals(
        faces_indexed: Int[Array, "faces vertices"],
        xyz: Float[Array, "vertices 3"],
    ) -> Float[Array, "faces 3"]:
        """
        Compute the unnormalized normal of every face in the mesh.

        Parameters
        ----------
        faces_indexed :
            The vertex indices of each face, with ``-1`` padding for absent
            vertices.
        xyz :
            The coordinates of the mesh vertices.

        Returns
        -------
        normals :
            The unnormalized normal of each face, its magnitude proportional to
            face area.
        """

        def face_normal(
            face: Int[Array, "vertices"],
            xyz: Float[Array, "vertices 3"],
        ) -> Float[Array, "3"]:
            face = jnp.ravel(face)
            xyz_face = xyz[face, :]
            xyz_repl = xyz_face[0, :]

            # NOTE: Replace -1 padding with the first vertex to avoid nans in the
            # gradient. The duplicated vertex adds a zero-area term to the Newell
            # sum, so the face normal is unaffected.
            fxyz = vmap(jnp.where, in_axes=(0, 0, None))(face >= 0, xyz_face, xyz_repl)

            return normal_polygon(fxyz, unitized=False)

        return vmap(face_normal, in_axes=(0, None))(faces_indexed, xyz)

    def vertex_normal(
        self,
        eq_state: EquilibriumState,
        faces_indexed: Int[Array, "faces vertices"],
        index: Int[Array, ""],
    ) -> Float[Array, "3"]:
        """
        The unitized normal at a vertex, averaged over its incident faces.

        Parameters
        ----------
        eq_state :
            The equilibrium state to read the vertex coordinates from.
        faces_indexed :
            The mesh face topology whose incident faces set the vertex normal.
        index :
            The index of the vertex.

        Returns
        -------
        normal :
            The unit normal at the vertex.

        Notes
        -----
        The incident faces are selected by masking the face topology directly:
        a face is incident when any of its vertex indices equals the queried
        index. Padding entries (``-1``) never match a valid index. The
        area-weighted face normals then sum into the vertex normal before
        normalizing.
        """
        face_normals = self.face_normals(faces_indexed, eq_state.xyz)
        mask = jnp.any(faces_indexed == index, axis=-1)
        normal = jnp.sum(jnp.reshape(mask, (-1, 1)) * face_normals, axis=0)

        return normalize_vector(normal)

    def prediction(
        self,
        eq_state: EquilibriumState,
        structure: EquilibriumMeshStructure,
        index: Int[Array, ""],
    ) -> Float[Array, ""]:
        """
        The angle between the vertex normal and the reference vector.

        Parameters
        ----------
        eq_state :
            The equilibrium state to read the vertex coordinates from.
        structure :
            The mesh structure providing the face topology.
        index :
            The index of the vertex.

        Returns
        -------
        prediction :
            The signed angle between the vertex normal and its reference vector, in
            radians.

        Notes
        -----
        The reference vector is read off `self.vector`, this element's own vector.
        """
        normal = self.vertex_normal(eq_state, structure.faces_indexed, index)

        # the signed angle is covariant with the normal's orientation, which
        # the winding of the incident faces determines
        return angle_vectors(normal, self.vector)

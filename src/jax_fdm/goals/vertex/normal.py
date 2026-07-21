from collections.abc import Sequence

import jax.numpy as jnp
import numpy as np
from jax import vmap
from jaxtyping import Array
from jaxtyping import Float
from jaxtyping import Int

from jax_fdm.equilibrium import EquilibriumMeshStructure
from jax_fdm.equilibrium import EquilibriumModel
from jax_fdm.equilibrium import EquilibriumState
from jax_fdm.geometry import angle_vectors
from jax_fdm.geometry import normal_polygon
from jax_fdm.geometry import normalize_vector
from jax_fdm.goals.goal import ScalarGoal
from jax_fdm.goals.goal import TargetLike
from jax_fdm.goals.vertex.vertex import VertexGoal

__all__ = ["VertexNormalAngleGoal"]


class VertexNormalAngleGoal(ScalarGoal, VertexGoal):
    """
    Drive the angle between a vertex normal and a reference vector toward a target.

    Parameters
    ----------
    key :
        The key of the vertex the goal acts on.
    vector :
        The reference vector each vertex normal's angle is measured against.
    target :
        The target angle, in radians.
    weight :
        The relative importance of the goal in the loss.

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

    def __init__(
        self,
        key: int,
        vector: Float[Array, "..."] | Sequence[float],
        target: TargetLike,
        weight: float = 1.0,
    ) -> None:
        super().__init__(key=key, target=target, weight=weight)
        self._vector: Float[Array, "vectors 3"]
        self.vector = vector

        # set in init() from the mesh structure, before any prediction runs
        self.faces_indexed: Int[Array, "faces vertices"]

    @property
    def vector(self) -> Float[Array, "vectors 3"]:
        """
        The reference vector each vertex normal's angle is measured against.
        """
        return self._vector

    @vector.setter
    def vector(self, vector: Float[Array, "..."] | Sequence[float]) -> None:
        self._vector = jnp.reshape(jnp.asarray(vector), (-1, 3))

    def vectors(self) -> Float[Array, "vectors 3"]:
        """
        Scatter the reference vectors into a per-index matrix.

        Returns
        -------
        vectors :
            A matrix holding each vertex's reference vector at its structure index.
        """
        matrix = np.zeros((max(self.index) + 1, 3))
        for vec, idx in zip(self.vector, self.index):
            matrix[idx, :] = vec
        return jnp.asarray(matrix)

    def init(
        self,
        model: EquilibriumModel,
        structure: EquilibriumMeshStructure,
    ) -> None:
        """
        Bind the goal to a mesh, caching face topology and reference vectors.

        Parameters
        ----------
        model :
            The equilibrium model.
        structure :
            The mesh structure providing face indices and face-vertex connectivity.
        """
        super().init(model, structure)
        self.vector = self.vectors()
        self.faces_indexed = structure.faces_indexed

    def face_normals(self, xyz: Float[Array, "nodes 3"]) -> Float[Array, "faces 3"]:
        """
        Compute the unnormalized normal of every face in the mesh.

        Parameters
        ----------
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
            xyz: Float[Array, "nodes 3"],
        ) -> Float[Array, "3"]:
            face = jnp.ravel(face)
            xyz_face = xyz[face, :]
            xyz_repl = xyz_face[0, :]

            # NOTE: Replace -1 padding with the first vertex to avoid nans in the
            # gradient. The duplicated vertex adds a zero-area term to the Newell
            # sum, so the face normal is unaffected.
            fxyz = vmap(jnp.where, in_axes=(0, 0, None))(face >= 0, xyz_face, xyz_repl)

            return normal_polygon(fxyz, unitized=False)

        return vmap(face_normal, in_axes=(0, None))(self.faces_indexed, xyz)

    def vertex_normal(
        self,
        eq_state: EquilibriumState,
        index: Int[Array, ""],
    ) -> Float[Array, "3"]:
        """
        The unitized normal at a vertex, averaged over its incident faces.

        Parameters
        ----------
        eq_state :
            The equilibrium state to read the vertex coordinates from.
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
        face_normals = self.face_normals(eq_state.xyz)
        mask = jnp.any(self.faces_indexed == index, axis=-1)
        normal = jnp.sum(jnp.reshape(mask, (-1, 1)) * face_normals, axis=0)

        return normalize_vector(normal)

    def prediction(
        self,
        eq_state: EquilibriumState,
        index: Int[Array, ""],
    ) -> Float[Array, ""]:
        """
        The angle between the vertex normal and the reference vector.

        Parameters
        ----------
        eq_state :
            The equilibrium state to read the vertex coordinates from.
        index :
            The index of the vertex.

        Returns
        -------
        prediction :
            The signed angle between the vertex normal and its reference vector, in
            radians.
        """
        normal = self.vertex_normal(eq_state, index)

        # the signed angle is covariant with the normal's orientation, which
        # the winding of the incident faces determines
        return angle_vectors(normal, self.vector[index, :])

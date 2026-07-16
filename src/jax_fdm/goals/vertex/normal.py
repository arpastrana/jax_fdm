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
from jax_fdm.goals import ScalarGoal
from jax_fdm.goals.vertex import VertexGoal


class VertexNormalAngleGoal(ScalarGoal, VertexGoal):
    """
    Reach a target value for the angle formed by the vertex normal and a reference vector.

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
        key: int | tuple[int, int] | list[int] | list[tuple[int, int]],
        vector: Float[Array, "..."],
        target: float | Float[Array, "..."],
        weight: float = 1.0,
    ) -> None:
        super().__init__(key=key, target=target, weight=weight)
        self._vector: Float[Array, "vectors 3"]
        self.vector = vector

        # set in init() from the mesh structure, before any prediction runs
        self.faces_indexed: Int[Array, "faces vertices"]
        self.connectivity_faces_vertices: Float[Array, "faces vertices"]

    @property
    def vector(self) -> Float[Array, "vectors 3"]:
        """
        The vector to take the angle with.
        """
        return self._vector

    @vector.setter
    def vector(self, vector: Float[Array, "..."]) -> None:
        self._vector = jnp.reshape(jnp.asarray(vector), (-1, 3))

    def vectors(self) -> Float[Array, "vectors 3"]:
        """
        Create a matrix of vectors.
        """
        matrix = np.zeros((max(self.index) + 1, 3))
        for vec, idx in zip(self.vector, self.index):
            matrix[idx, :] = vec
        return jnp.asarray(matrix)

    def init(self, model: EquilibriumModel, structure: EquilibriumMeshStructure) -> None:
        """
        Initialize the goal with information from an equilibrium model.
        """
        super().init(model, structure)
        self.vector = self.vectors()
        self.faces_indexed = structure.faces_indexed
        self.connectivity_faces_vertices = structure.connectivity_faces_vertices

    def face_normals(self, xyz: Float[Array, "nodes 3"]) -> Float[Array, "faces 3"]:
        """
        Compute the (unnormalized) normal of every face in the mesh.
        """
        def face_normal(face: Int[Array, "vertices"], xyz: Float[Array, "nodes 3"]) -> Float[Array, "3"]:
            face = jnp.ravel(face)
            xyz_face = xyz[face, :]
            xyz_repl = xyz_face[0, :]

            # NOTE: Replace -1 padding with the first vertex to avoid nans in the
            # gradient. The duplicated vertex adds a zero-area term to the Newell
            # sum, so the face normal is unaffected.
            fxyz = vmap(jnp.where, in_axes=(0, 0, None))(face >= 0, xyz_face, xyz_repl)

            return normal_polygon(fxyz, unitized=False)

        return vmap(face_normal, in_axes=(0, None))(self.faces_indexed, xyz)

    def vertex_normal(self, eq_state: EquilibriumState, index: Int[Array, ""]) -> Float[Array, "3"]:
        """
        Get the unitized normal vector at a vertex.
        """
        face_normals = self.face_normals(eq_state.xyz)
        mask = jnp.where(self.connectivity_faces_vertices[:, index] > 0.0, 1.0, 0.0)
        normal = jnp.sum(jnp.reshape(mask, (-1, 1)) * face_normals, axis=0)

        return normalize_vector(normal)

    def prediction(self, eq_state: EquilibriumState, index: Int[Array, ""]) -> Float[Array, "1"]:
        """
        Returns the angle between the vertex normal and the reference vector.
        """
        normal = self.vertex_normal(eq_state, index)

        # the signed angle is covariant with the normal's orientation, which
        # the winding of the incident faces determines
        angle = angle_vectors(normal, self.vector[index, :])

        return jnp.atleast_1d(angle)

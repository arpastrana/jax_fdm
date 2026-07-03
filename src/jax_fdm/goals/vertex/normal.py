import jax.numpy as jnp
import numpy as np
from jax import vmap

from jax_fdm.geometry import cosine_vectors
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

    The angle is measured from the absolute cosine, so it lies in [0, pi / 2]
    and is invariant to the vertex normal's orientation. An averaged vertex
    normal flips sign with the winding of its incident faces, which is an
    artifact of how the mesh was built rather than a property of the surface;
    folding the angle into [0, pi / 2] keeps the goal from chasing that sign.
    """
    def __init__(self, key, vector, target, weight=1.0):
        super().__init__(key=key, target=target, weight=weight)
        self._vector = None
        self.vector = vector

        self.faces_indexed = None
        self.connectivity_faces_vertices = None

    @property
    def vector(self):
        """
        The vector to take the angle with.
        """
        return self._vector

    @vector.setter
    def vector(self, vector):
        self._vector = jnp.reshape(jnp.asarray(vector), (-1, 3))

    def vectors(self):
        """
        Create a matrix of vectors.
        """
        matrix = np.zeros((max(self.index) + 1, 3))
        for vec, idx in zip(self.vector, self.index):
            matrix[idx, :] = vec
        return matrix

    def init(self, model, structure):
        """
        Initialize the goal with information from an equilibrium model.
        """
        super().init(model, structure)
        self.vector = self.vectors()
        self.faces_indexed = structure.faces_indexed
        self.connectivity_faces_vertices = structure.connectivity_faces_vertices

    def face_normals(self, xyz):
        """
        Compute the (unnormalized) normal of every face in the mesh.
        """
        def face_normal(face, xyz):
            face = jnp.ravel(face)
            xyz_face = xyz[face, :]
            xyz_repl = xyz_face[0, :]

            # NOTE: Replace -1 padding with the first vertex to avoid nans in the
            # gradient. The duplicated vertex adds a zero-area term to the Newell
            # sum, so the face normal is unaffected.
            fxyz = vmap(jnp.where, in_axes=(0, 0, None))(face >= 0, xyz_face, xyz_repl)

            return normal_polygon(fxyz, unitized=False)

        return vmap(face_normal, in_axes=(0, None))(self.faces_indexed, xyz)

    def vertex_normal(self, eqstate, index):
        """
        Get the unitized normal vector at a vertex.
        """
        face_normals = self.face_normals(eqstate.xyz)
        mask = jnp.where(self.connectivity_faces_vertices[:, index] > 0.0, 1.0, 0.0)
        normal = jnp.sum(jnp.reshape(mask, (-1, 1)) * face_normals, axis=0)

        return normalize_vector(normal)

    def prediction(self, eqstate, index):
        """
        Returns the angle between the vertex normal and the reference vector.
        """
        normal = self.vertex_normal(eqstate, index)

        # absolute cosine folds the angle into [0, pi / 2] so the orientation of
        # the averaged vertex normal (a winding artifact) does not matter. Clip
        # guards the arccos gradient, which is singular when the vectors align.
        cosine = jnp.abs(cosine_vectors(normal, self.vector[index, :]))
        angle = jnp.arccos(jnp.clip(cosine, -1.0, 1.0))

        return jnp.atleast_1d(angle)

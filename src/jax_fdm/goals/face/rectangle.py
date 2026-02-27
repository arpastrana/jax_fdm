from jax import vmap

import jax.numpy as jnp

from jax_fdm.goals import ScalarGoal
from jax_fdm.goals.face import FaceGoal

from jax_fdm.geometry import cosines_angles_polygon


class FaceRectangleGoal(ScalarGoal, FaceGoal):
    """
    Make the internal angles of a quadrilateral mesh face reach 90 degrees.

    Notes
    -----
    This goal is only applicable to quadrilateral mesh faces.
    """
    def __init__(self, key, weight=1.0, target=0.0):
        super().__init__(key=key, target=target, weight=weight)
        self.face_indices = None

    def init(self, model, structure):
        """
        Initialize the goal with information of a model and a structure.
        """
        super().init(model, structure)
        face_indices = structure.faces_indexed[self.index]
        self.face_indices = face_indices[:, :4]

    def prediction(self, eq_state, index):
        """
        The sum of the cosine of the internal angles of a face.
        """
        fxyz = eq_state.xyz[self.face_indices]
        face_cosines = vmap(cosines_angles_polygon, in_axes=(0))(fxyz)

        return jnp.atleast_1d(jnp.sum(jnp.abs(face_cosines)))

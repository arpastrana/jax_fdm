import jax.numpy as jnp
from jaxtyping import Array
from jaxtyping import Float
from jaxtyping import Int

from jax_fdm.equilibrium import EquilibriumMeshStructure
from jax_fdm.equilibrium import EquilibriumState
from jax_fdm.geometry import cosines_angles_polygon
from jax_fdm.goals.face.face import FaceGoal
from jax_fdm.goals.goal import ScalarGoal
from jax_fdm.goals.goal import TargetLike

__all__ = ["FaceRectangularGoal"]


class FaceRectangularGoal(ScalarGoal, FaceGoal):
    """
    Make the internal angles of a quadrilateral mesh face reach 90 degrees.

    Notes
    -----
    This goal is only applicable to quadrilateral mesh faces.
    """

    def __init__(
        self,
        key: int,
        weight: float = 1.0,
        target: TargetLike = 0.0,
    ) -> None:
        super().__init__(key=key, target=target, weight=weight)

    def prediction(
        self,
        eq_state: EquilibriumState,
        structure: EquilibriumMeshStructure,
        index: Int[Array, ""],
    ) -> Float[Array, ""]:
        """
        A measure of how far a face is from rectangular.

        Parameters
        ----------
        eq_state :
            The equilibrium state to read the face coordinates from.
        structure :
            The mesh structure providing the face topology.
        index :
            The index of the face.

        Returns
        -------
        prediction :
            The mean absolute cosine of the face's corner angles, zero when
            every corner is a right angle.
        """
        fxyz = eq_state.xyz[structure.faces_indexed[index, :4]]
        face_cosines = cosines_angles_polygon(fxyz)

        return jnp.mean(jnp.abs(face_cosines))

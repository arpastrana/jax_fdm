import numpy as np
from jaxtyping import Array
from jaxtyping import Float
from jaxtyping import Int

from jax_fdm.equilibrium import EquilibriumState
from jax_fdm.goals.vertex import VertexNormalAngleGoal


class VertexTangentAngleGoal(VertexNormalAngleGoal):
    """
    Reach a target value for the angle formed by the vertex tangent and a reference vector.

    The tangent angle is calculated as 90 degrees minus the vertex normal
    angle, so it is signed and spans [-pi / 2, pi / 2]: positive when the
    vertex normal points within 90 degrees of the reference vector (the
    surface rises toward it) and negative when the normal is folded away.
    The sign follows the winding of the mesh faces, which must be unified;
    see the notes of `VertexNormalAngleGoal`.
    """
    def __init__(
        self,
        key: int | tuple[int, int] | list[int] | list[tuple[int, int]],
        vector: Float[Array, "..."],
        target: float | Float[Array, "..."],
        weight: float = 1.0,
    ) -> None:
        super().__init__(key=key, vector=vector, target=target, weight=weight)

    def prediction(self, eq_state: EquilibriumState, index: Int[Array, ""]) -> Float[Array, "1"]:
        """
        Returns the angle between the vertex tangent and the reference vector.
        """
        angle_normal = super().prediction(eq_state, index)

        angle_tangent = np.pi * 0.5 - angle_normal

        return angle_tangent

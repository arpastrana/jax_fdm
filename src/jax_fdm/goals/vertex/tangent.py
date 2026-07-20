from collections.abc import Sequence

import numpy as np
from jaxtyping import Array
from jaxtyping import Float
from jaxtyping import Int

from jax_fdm.equilibrium import EquilibriumState
from jax_fdm.goals.goal import TargetLike
from jax_fdm.goals.vertex import VertexNormalAngleGoal


class VertexTangentAngleGoal(VertexNormalAngleGoal):
    """
    Drive the angle between a vertex tangent and a reference vector toward a target.

    Parameters
    ----------
    key :
        The key of the vertex the goal acts on.
    vector :
        The reference vector each vertex tangent's angle is measured against.
    target :
        The target angle, in radians.
    weight :
        The relative importance of the goal in the loss.

    Notes
    -----
    The tangent angle is 90 degrees minus the vertex normal angle, so it is signed
    and spans [-pi / 2, pi / 2]: positive when the vertex normal points within 90
    degrees of the reference vector (the surface rises toward it) and negative when
    the normal is folded away. The sign follows the winding of the mesh faces, which
    must be unified; see the notes of
    [VertexNormalAngleGoal][jax_fdm.goals.vertex.normal.VertexNormalAngleGoal].
    """

    def __init__(
        self,
        key: int,
        vector: Float[Array, "..."] | Sequence[float],
        target: TargetLike,
        weight: float = 1.0,
    ) -> None:
        super().__init__(key=key, vector=vector, target=target, weight=weight)

    def prediction(
        self,
        eq_state: EquilibriumState,
        index: Int[Array, ""],
    ) -> Float[Array, "1"]:
        """
        The angle between the vertex tangent and the reference vector.

        Parameters
        ----------
        eq_state :
            The equilibrium state to read the vertex coordinates from.
        index :
            The index of the vertex.

        Returns
        -------
        prediction :
            The signed tangent angle, 90 degrees minus the vertex normal angle, in
            radians.
        """
        angle_normal = super().prediction(eq_state, index)

        angle_tangent = np.pi * 0.5 - angle_normal

        return angle_tangent

from collections.abc import Sequence

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array
from jaxtyping import Float
from jaxtyping import Int

from jax_fdm.equilibrium import EquilibriumState
from jax_fdm.equilibrium import EquilibriumStructure
from jax_fdm.geometry import angle_vectors
from jax_fdm.goals.edge.edge import EdgeGoal
from jax_fdm.goals.goal import TargetLike

__all__ = ["EdgeAngleGoal"]


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


class EdgeAngleGoal(EdgeGoal):
    """
    Drive the angle between an edge and a reference vector toward a target.

    Parameters
    ----------
    key :
        The key of the edge the goal acts on.
    target :
        The target angle, in radians.
    weight :
        The relative importance of the goal in the loss.
    vector :
        The reference vector the edge's angle is measured against. Keyword-only
        and required, since there is no meaningful default reference direction.
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
        key: tuple[int, int],
        target: TargetLike,
        weight: float = 1.0,
        *,
        vector: Float[Array, "..."] | Sequence[float],
    ) -> None:
        self.key = key
        self.target = target
        self.weight = weight
        self.vector = _as_vector(vector)

    def prediction(
        self,
        eq_state: EquilibriumState,
        structure: EquilibriumStructure,
        index: Int[Array, ""],
    ) -> Float[Array, ""]:
        """
        The angle between the edge and its reference vector.

        Parameters
        ----------
        eq_state :
            The equilibrium state to read the edge vector from.
        structure :
            The structure the goal is evaluated against; unused.
        index :
            The index of the edge.

        Returns
        -------
        prediction :
            The angle between the edge and its reference vector, in radians.

        Notes
        -----
        The reference vector is read off `self.vector`, this element's own vector.
        """
        return angle_vectors(eq_state.vectors[index, :], self.vector)

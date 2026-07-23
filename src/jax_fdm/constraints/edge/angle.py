from collections.abc import Sequence

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array
from jaxtyping import Float
from jaxtyping import Int

from jax_fdm.constraints.constraint import BoundLike
from jax_fdm.constraints.edge.edge import EdgeConstraint
from jax_fdm.equilibrium import EquilibriumState
from jax_fdm.equilibrium import EquilibriumStructure
from jax_fdm.geometry import angle_vectors

__all__ = ["EdgeAngleConstraint"]


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
        The reference vector as a JAX array, unbatched like the constraint's
        other leaves.
    """
    return jnp.asarray(vector)


class EdgeAngleConstraint(EdgeConstraint):
    """
    Bound the angle between an edge and a reference vector.

    Parameters
    ----------
    key :
        The key of the edge the constraint acts on.
    vector :
        The reference vector the edge's angle is measured against.
    bound_low :
        The lower bound on the angle, in radians. If None, unbounded below.
    bound_up :
        The upper bound on the angle, in radians. If None, unbounded above.
    """

    # kw_only lets this required leaf follow the base's defaulted bounds (the
    # dataclass "non-default after default" rule) without a default of its own.
    # A normal init field (not init=False), so equinox does not warn about a
    # differentiable leaf; the custom __init__ below sets and coerces it. Stored
    # unbatched like the constraint's other leaves, so it is a plain (3,) vector
    # this element's constraint reads directly.
    vector: Float[Array, "3"] = eqx.field(kw_only=True)

    def __init__(
        self,
        key: tuple[int, int],
        vector: Float[Array, "..."] | Sequence[float],
        bound_low: BoundLike = None,
        bound_up: BoundLike = None,
    ) -> None:
        self.key = key
        self.vector = _as_vector(vector)
        self.bound_low = bound_low
        self.bound_up = bound_up

    def constraint(
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
            The structure the constraint is evaluated against; unused.
        index :
            The index of the edge.

        Returns
        -------
        constraint :
            The angle between the edge and its reference vector, in radians.

        Notes
        -----
        The reference vector is read off `self.vector`, this element's own vector.
        """
        return angle_vectors(eq_state.vectors[index, :], self.vector)

from collections.abc import Sequence

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array
from jaxtyping import Float
from jaxtyping import Int

from jax_fdm.constraints.edge.edge import EdgeConstraint
from jax_fdm.equilibrium import EquilibriumState
from jax_fdm.equilibrium import EquilibriumStructure
from jax_fdm.geometry import angle_vectors

__all__ = ["EdgeAngleConstraint"]


class EdgeAngleConstraint(EdgeConstraint):
    """
    Bound the angle between an edge and a reference vector.

    Parameters
    ----------
    key :
        The key or keys of the edge(s) the constraint acts on.
    vector :
        The reference vector each edge's angle is measured against.
    bound_low :
        The lower bound on the angle, in radians. If None, unbounded below.
    bound_up :
        The upper bound on the angle, in radians. If None, unbounded above.
    """

    def __init__(
        self,
        key: tuple[int, int] | list[tuple[int, int]],
        vector: Float[Array, "..."] | Sequence[float],
        bound_low: float | Float[Array, "..."] | None,
        bound_up: float | Float[Array, "..."] | None,
    ) -> None:
        super().__init__(key=key, bound_low=bound_low, bound_up=bound_up)
        self._vector: Float[Array, "vectors 3"]
        self.vector = vector

    @property
    def vector(self) -> Float[Array, "vectors 3"]:
        """
        The reference vector each edge's angle is measured against.
        """
        return self._vector

    @vector.setter
    def vector(self, vector: Float[Array, "..."] | Sequence[float]) -> None:
        self._vector = jnp.reshape(jnp.asarray(vector), (-1, 3))

    def operand(
        self,
        structure: EquilibriumStructure,
    ) -> tuple[Int[np.ndarray, "edges"], Float[Array, "edges 3"]]:
        """
        The per-element payload: the edge indices paired with reference vectors.

        Parameters
        ----------
        structure :
            The structure whose edge ordering defines the indices.

        Returns
        -------
        payload :
            The edge index array and the reference vectors, both collection
            ordered, so vmap zips each edge's index with its own reference vector.
        """
        return self.indices(structure), self.vector

    def constraint(
        self,
        eq_state: EquilibriumState,
        structure: EquilibriumStructure,
        payload: tuple[Float[Array, ""], Float[Array, "3"]],
    ) -> Float[Array, ""]:
        """
        The angle between the edge and its reference vector.

        Parameters
        ----------
        eq_state :
            The equilibrium state to read the edge vector from.
        structure :
            The structure the constraint is evaluated against; unused.
        payload :
            The edge index and its reference vector for this element.

        Returns
        -------
        constraint :
            The angle between the edge and its reference vector, in radians.
        """
        index, vector = payload

        return angle_vectors(eq_state.vectors[index, :], vector)

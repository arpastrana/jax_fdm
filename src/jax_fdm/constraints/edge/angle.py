import jax.numpy as jnp
import numpy as np
from jaxtyping import Array
from jaxtyping import Float
from jaxtyping import Int

from jax_fdm.constraints.edge import EdgeConstraint
from jax_fdm.equilibrium import EquilibriumModel
from jax_fdm.equilibrium import EquilibriumState
from jax_fdm.equilibrium import EquilibriumStructure
from jax_fdm.geometry import angle_vectors


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
        vector: Float[Array, "..."],
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
    def vector(self, vector: Float[Array, "..."]) -> None:
        self._vector = jnp.reshape(jnp.asarray(vector), (-1, 3))

    def vectors(self) -> Float[Array, "vectors 3"]:
        """
        Scatter the reference vectors into a per-index matrix.

        Returns
        -------
        vectors :
            A matrix holding each edge's reference vector at its structure index.
        """
        matrix = np.zeros((max(self.index) + 1, 3))
        for vec, idx in zip(self.vector, self.index):
            matrix[idx, :] = vec
        return jnp.asarray(matrix)

    def init(self, model: EquilibriumModel, structure: EquilibriumStructure) -> None:
        """
        Bind the constraint to a structure and reindex its reference vectors.

        Parameters
        ----------
        model :
            The equilibrium model.
        structure :
            The structure whose edge ordering defines the indices.
        """
        super().init(model, structure)
        self.vector = self.vectors()

    def constraint(
        self,
        eq_state: EquilibriumState,
        index: Int[Array, ""],
    ) -> Float[Array, ""]:
        """
        The angle between the edge and its reference vector.

        Parameters
        ----------
        eq_state :
            The equilibrium state to read the edge vector from.
        index :
            The index of the edge.

        Returns
        -------
        constraint :
            The angle between the edge and its reference vector, in radians.
        """
        vector = eq_state.vectors[index, :]
        return angle_vectors(vector, self.vector[index, :])

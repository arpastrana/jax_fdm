import jax.numpy as jnp
import numpy as np
from jaxtyping import Array
from jaxtyping import Float
from jaxtyping import Int

from jax_fdm.equilibrium import EquilibriumModel
from jax_fdm.equilibrium import EquilibriumState
from jax_fdm.equilibrium import EquilibriumStructure
from jax_fdm.geometry import angle_vectors
from jax_fdm.goals import ScalarGoal
from jax_fdm.goals.edge import EdgeGoal


class EdgeAngleGoal(ScalarGoal, EdgeGoal):
    """
    Reach a target angle between the direction of an edge and a reference vector.
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

    def init(self, model: EquilibriumModel, structure: EquilibriumStructure) -> None:
        """
        Initialize the goal with information from an equilibrium model.
        """
        super().init(model, structure)
        self.vector = self.vectors()

    def prediction(
        self,
        eq_state: EquilibriumState,
        index: Int[Array, ""],
    ) -> Float[Array, "1"]:
        """
        The angle between the edge and the reference vector.
        """
        vector = eq_state.vectors[index, :]

        return jnp.atleast_1d(angle_vectors(vector, self.vector[index, :]))

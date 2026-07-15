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
        key: int | tuple[int, int] | list,
        vector: Float[Array, "..."],
        target: float | Float[Array, "..."],
        weight: float = 1.0,
    ):
        super().__init__(key=key, target=target, weight=weight)
        self._vector: Float[Array, "vectors 3"] | None = None
        self.vector = vector

    @property
    def vector(self) -> Float[Array, "vectors 3"] | None:
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
        matrix = np.zeros((max(self.index) + 1, 3))  # pyright: ignore[reportArgumentType]  # self.index is Optional by declaration but always populated by init() before this runs
        for vec, idx in zip(self.vector, self.index):  # pyright: ignore[reportArgumentType]  # self.vector/self.index are always arrays by the time vectors() runs
            matrix[idx, :] = vec
        return jnp.asarray(matrix)

    def init(self, model: EquilibriumModel, structure: EquilibriumStructure) -> None:
        """
        Initialize the goal with information from an equilibrium model.
        """
        super().init(model, structure)
        self.vector = self.vectors()

    def prediction(self, eq_state: EquilibriumState, index: Int[Array, ""]) -> Float[Array, "1"]:
        """
        The angle between the edge and the reference vector.
        """
        vector = eq_state.vectors[index, :]

        return jnp.atleast_1d(angle_vectors(vector, self.vector[index, :]))  # pyright: ignore[reportOptionalSubscript]  # self.vector is Optional by declaration but always set in __init__ before this runs

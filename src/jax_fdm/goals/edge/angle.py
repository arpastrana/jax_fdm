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
        vector: Float[Array, "..."] | Float[np.ndarray, "..."],
        target: float | Float[Array, "..."],
        weight: float = 1.0,
    ):
        super().__init__(key=key, target=target, weight=weight)
        self._vector = None
        self.vector = vector

    @property
    def vector(self):
        """
        The vector to take the angle with.
        """
        return self._vector

    @vector.setter
    def vector(self, vector: Float[Array, "..."] | Float[np.ndarray, "..."]) -> None:
        self._vector = jnp.reshape(jnp.asarray(vector), (-1, 3))

    def init(self, model: EquilibriumModel, structure: EquilibriumStructure) -> None:
        """
        Initialize the goal with information from an equilibrium model.
        """
        super().init(model, structure)

        # create matrix of vectors
        vector = self.vector
        vm = np.zeros((max(self.index) + 1, 3))  # pyright: ignore[reportArgumentType]  # self.index is Optional by declaration but always populated by init() before this runs
        for v, idx in zip(vector, self.index):  # pyright: ignore[reportArgumentType]  # self.index is Optional by declaration but always populated by init() before this runs
            vm[idx, :] = v
        self.vector = vm

    def prediction(self, eq_state: EquilibriumState, index: Int[Array, ""]) -> Float[Array, "1"]:
        """
        The angle between the edge and the reference vector.
        """
        vector = eq_state.vectors[index, :]

        return jnp.atleast_1d(angle_vectors(vector, self.vector[index, :]))  # pyright: ignore[reportOptionalSubscript]  # self.vector is Optional by declaration but always set in __init__ before this runs

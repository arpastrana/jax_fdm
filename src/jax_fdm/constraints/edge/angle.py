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
    Constraints the angle formed by an edge and a vector between a lower and an upper bound.
    """
    def __init__(
        self,
        key: int | tuple[int, int] | list[int] | list[tuple[int, int]],
        vector: Float[Array, "..."] | Float[np.ndarray, "..."],
        bound_low: float | Float[Array, "..."] | None,
        bound_up: float | Float[Array, "..."] | None,
    ):
        super().__init__(key=key, bound_low=bound_low, bound_up=bound_up)
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

    def vectors(self) -> Float[np.ndarray, "elements 3"]:
        """
        Create a matrix of vectors.
        """
        matrix = np.zeros((max(self.index) + 1, 3))  # pyright: ignore[reportArgumentType]  # self.index is Optional by declaration but always populated by init() before this runs
        for vec, idx in zip(self.vector, self.index):  # pyright: ignore[reportArgumentType]  # self.vector/self.index are always arrays by the time vectors() runs
            matrix[idx, :] = vec
        return matrix

    def init(self, model: EquilibriumModel, structure: EquilibriumStructure) -> None:
        """
        Initialize the constraint with information from an equilibrium model.
        """
        super().init(model, structure)
        self.vector = self.vectors()

    def constraint(self, eqstate: EquilibriumState, index: Int[Array, ""]) -> Float[Array, ""]:
        """
        Returns the angle between an edge in an equilibrium state and a vector.
        """
        vector = eqstate.vectors[index, :]
        return angle_vectors(vector, self.vector[index, :])  # pyright: ignore[reportOptionalSubscript]  # self.vector is Optional by declaration but always set in __init__ before this runs

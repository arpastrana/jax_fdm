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
    Drive the angle between an edge and a reference vector toward a target.

    Parameters
    ----------
    key :
        The key or keys of the edge(s) the goal acts on.
    vector :
        The reference vector each edge's angle is measured against.
    target :
        The target angle, in radians.
    weight :
        The relative importance of the goal in the loss.
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

        Notes
        -----
        Rebuilds the reference vectors indexed by structure position so that
        :meth:`prediction` can gather them with the same index as the edges.
        """
        matrix = np.zeros((max(self.index) + 1, 3))
        for vec, idx in zip(self.vector, self.index):
            matrix[idx, :] = vec
        return jnp.asarray(matrix)

    def init(self, model: EquilibriumModel, structure: EquilibriumStructure) -> None:
        """
        Bind the goal to a structure and reindex its reference vectors.

        Parameters
        ----------
        model :
            The equilibrium model.
        structure :
            The structure whose edge ordering defines the indices.
        """
        super().init(model, structure)
        self.vector = self.vectors()

    def prediction(
        self,
        eq_state: EquilibriumState,
        index: Int[Array, ""],
    ) -> Float[Array, "1"]:
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
        prediction :
            The angle between the edge and its reference vector, in radians.
        """
        vector = eq_state.vectors[index, :]

        return jnp.atleast_1d(angle_vectors(vector, self.vector[index, :]))

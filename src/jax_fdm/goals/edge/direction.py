from jaxtyping import Array
from jaxtyping import Float
from jaxtyping import Int

from jax_fdm.equilibrium import EquilibriumState
from jax_fdm.geometry import normalize_vector
from jax_fdm.goals import VectorGoal
from jax_fdm.goals.edge import EdgeGoal


class EdgeDirectionGoal(VectorGoal, EdgeGoal):
    """
    Align an edge's direction with a target vector.

    Notes
    -----
    Both the prediction and target are unit-normalized, so only direction is
    compared and edge length is ignored.
    """

    def prediction(
        self,
        eq_state: EquilibriumState,
        index: Int[Array, ""],
    ) -> Float[Array, "3"]:
        """
        The unit direction of the edge.

        Parameters
        ----------
        eq_state :
            The equilibrium state to read the edge vector from.
        index :
            The index of the edge.

        Returns
        -------
        prediction :
            The normalized edge vector.
        """
        vector = eq_state.vectors[index, :]
        return normalize_vector(vector)

    def goal(
        self,
        target: Float[Array, "3"],
        prediction: Float[Array, "3"],
    ) -> Float[Array, "3"]:
        """
        The unit direction of the target vector.

        Parameters
        ----------
        target :
            The target direction vector.
        prediction :
            The current normalized edge vector, unused.

        Returns
        -------
        goal :
            The normalized target vector.
        """
        return normalize_vector(target)

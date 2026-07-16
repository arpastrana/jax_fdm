"""Goals defined on the residual forces at network nodes."""

from jaxtyping import Array
from jaxtyping import Float
from jaxtyping import Int

from jax_fdm.equilibrium import EquilibriumState
from jax_fdm.geometry import length_vector
from jax_fdm.geometry import normalize_vector
from jax_fdm.goals import ScalarGoal
from jax_fdm.goals import VectorGoal
from jax_fdm.goals.node import NodeGoal


class NodeResidualForceGoal(ScalarGoal, NodeGoal):
    """
    Drive the residual force magnitude at a node toward a target value.
    """

    def prediction(
        self,
        eq_state: EquilibriumState,
        index: Int[Array, ""],
    ) -> Float[Array, "1"]:
        """
        The magnitude of the residual force at the node.

        Parameters
        ----------
        eq_state :
            The equilibrium state to read the residual from.
        index :
            The index of the node.

        Returns
        -------
        prediction :
            The Euclidean magnitude of the node's residual force.
        """
        residual = eq_state.residuals[index, :]

        return length_vector(residual)


class NodeResidualVectorGoal(VectorGoal, NodeGoal):
    """
    Drive the residual force at a node toward a target vector.
    """

    def prediction(
        self,
        eq_state: EquilibriumState,
        index: Int[Array, ""],
    ) -> Float[Array, "3"]:
        """
        The residual force vector at the node.

        Parameters
        ----------
        eq_state :
            The equilibrium state to read the residual from.
        index :
            The index of the node.

        Returns
        -------
        prediction :
            The node's residual force vector.
        """
        return eq_state.residuals[index, :]


class NodeResidualDirectionGoal(VectorGoal, NodeGoal):
    """
    Drive the residual force at a node toward a target direction.

    Notes
    -----
    Both the prediction and the target are unit-normalized, so only direction is
    compared while magnitude is ignored. Euclidean distance between the normalized
    vectors gives the same ordering as cosine distance, without the cost of
    trigonometric operations.
    """

    def prediction(
        self,
        eq_state: EquilibriumState,
        index: Int[Array, ""],
    ) -> Float[Array, "3"]:
        """
        The unit direction of the residual force at the node.

        Parameters
        ----------
        eq_state :
            The equilibrium state to read the residual from.
        index :
            The index of the node.

        Returns
        -------
        prediction :
            The normalized residual force vector.
        """
        residual = eq_state.residuals[index, :]

        return normalize_vector(residual)

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
            The current normalized residual, unused.

        Returns
        -------
        goal :
            The normalized target vector.
        """
        return normalize_vector(target)

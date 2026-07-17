"""Goals defined on the residual forces at network nodes."""

import jax.numpy as jnp
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


class NodeResidualPlaneGoal(VectorGoal, NodeGoal):
    """
    Drive the residual vector at a node to lie in a target plane.

    Notes
    -----
    The residual is a vector, not a point, so the target plane passes through
    the origin and is described by its normal vector alone. The normal is
    unit-normalized internally, as the projection formula is only correct for
    a unit normal, so the input normal does not have to be unit-length.

    The goal returns the residual direction's projection onto the plane, which
    is the closest in-plane vector to the current prediction. The error term
    ``prediction - goal`` is then the out-of-plane component of the residual
    direction, which vanishes exactly when the residual lies in the plane.

    The prediction is unit-normalized too so that the error depends on direction
    alone: it measures the sine of the angle between the residual vector and the
    plane, and cannot be reduced by shrinking the reaction magnitude instead
    of rotating the reaction into the plane.
    """

    def prediction(
        self,
        eq_state: EquilibriumState,
        index: Int[Array, ""],
    ) -> Float[Array, "3"]:
        """
        The unit direction of the residual vector at the node.

        Parameters
        ----------
        eq_state :
            The equilibrium state to read the residual from.
        index :
            The index of the node.

        Returns
        -------
        prediction :
            The normalized residual vector.
        """
        residual = eq_state.residuals[index, :]

        return normalize_vector(residual)

    def goal(
        self,
        target: Float[Array, "3"],
        prediction: Float[Array, "3"],
    ) -> Float[Array, "3"]:
        """
        The projection of the predicted residual direction onto the target plane.

        Parameters
        ----------
        target :
            The target plane's normal vector.
        prediction :
            The current normalized residual direction.

        Returns
        -------
        goal :
            The prediction projected onto the target plane.
        """
        normal = normalize_vector(target)

        return prediction - jnp.dot(prediction, normal) * normal

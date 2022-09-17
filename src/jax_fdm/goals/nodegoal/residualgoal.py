"""
A bunch of goals to strive for.
"""
import numpy as np

import jax.numpy as jnp

from jax_fdm.goals import ScalarGoal
from jax_fdm.goals import VectorGoal

from jax_fdm.goals.nodegoal import NodeGoal


class NodeResidualForceGoal(ScalarGoal, NodeGoal):
    """
    Make the residual force in a network to match a non-negative magnitude.
    """
    def __init__(self, key, target, weight=1.0):
        # assert target >= 0.0, "Only non-negative target forces are supported!"
        super().__init__(key=key, target=target, weight=weight)

    def prediction(self, eq_state):
        """
        The residual at the the predicted node of the network.
        """
        residual = eq_state.residuals[self.index, :]
        return jnp.atleast_1d(jnp.linalg.norm(residual, axis=-1, keepdims=True))


class NodeResidualVectorGoal(VectorGoal, NodeGoal):
    """
    Make the residual force in a network to match the magnitude and direction of a vector.
    """
    def __init__(self, key, target, weight=1.0):
        super().__init__(key=key, target=target, weight=weight)

    def prediction(self, eq_state):
        """
        The residual at the the predicted node of the network.
        """
        return eq_state.residuals[self.index, :]


class NodeResidualDirectionGoal(VectorGoal, NodeGoal):
    """
    Make the residual force in a network to match the direction of a vector.

    Another effective proxy for cosine distance can be obtained by
    L2 normalisation of the vectors, followed by the application of normal
    Euclidean distance. Using this technique each term in each vector is
    first divided by the magnitude of the vector, yielding a vector of unit
    length. Then, it is clear, the Euclidean distance over the end-points
    of any two vectors is a proper metric which gives the same ordering as
    the cosine distance (a monotonic transformation of Euclidean distance;
    see below) for any comparison of vectors, and furthermore avoids the
    potentially expensive trigonometric operations required to yield
    a proper metric.
    """
    def __init__(self, key, target, weight=1.0):
        super().__init__(key=key, target=target, weight=weight)

    def prediction(self, eq_state):
        """
        The residual at the the predicted node of the network.
        """
        residual = eq_state.residuals[self.index, :]
        return residual / jnp.linalg.norm(residual, axis=-1, keepdims=True)  # unitized residual

    def target(self, prediction):
        """
        """
        target = np.array(self._target)
        return target / np.linalg.norm(target, axis=-1, keep_dims=True)

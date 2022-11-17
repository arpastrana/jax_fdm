"""
A bunch of goals to strive for.
"""
from jax_fdm.geometry import length_vector
from jax_fdm.geometry import normalize_vector

from jax_fdm.goals import ScalarGoal
from jax_fdm.goals import VectorGoal

from jax_fdm.goals.node import NodeGoal


class NodeResidualForceGoal(ScalarGoal, NodeGoal):
    """
    Make the residual force in a network to match a non-negative magnitude.
    """
    @staticmethod
    def prediction(eq_state, index):
        """
        The residual at the the predicted node of the network.
        """
        residual = eq_state.residuals[index, :]

        return length_vector(residual)


class NodeResidualVectorGoal(VectorGoal, NodeGoal):
    """
    Make the residual force in a network to match the magnitude and direction of a vector.
    """
    @staticmethod
    def prediction(eq_state, index):
        """
        The residual at the the predicted node of the network.
        """
        return eq_state.residuals[index, :]


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
    @staticmethod
    def prediction(eq_state, index):
        """
        The residual at the the predicted node of the network.
        """
        residual = eq_state.residuals[index, :]

        return normalize_vector(residual)

    @staticmethod
    def goal(target, prediction):
        """
        """
        return normalize_vector(target)

import jax.numpy as jnp

from jax_fdm.goals import ScalarGoal

from jax_fdm.goals.network import NetworkGoal


class NetworkXYZLaplacianGoal(ScalarGoal, NetworkGoal):
    """
    Minimize the Laplacian energy of the XYZ coordinates of a network.

    Notes
    -----
    This goal can be handy to "smoothen" the looks of a network.

    The Laplacian energy of some feature vector x is defined as
    E = x^T @ L @ x, where the unnormalized Laplacian L is related to
    the connectivity matrix of a network C such that L = C^T @ C.

    Given the above relationship and that we want to calculate the
    energy of the XYZ coordinates of a network in equilibrium
    (the feature vector x of interest are the XYZ coordinates),
    then the energy expression becomes: x^T @ (C^T @ C) @ X.

    We know that U = C @ X is the matrix with coordinate difference
    vectors that we obtain from the equilibrium model.

    Therefore, we can re-rexpress the energy E in terms of U, and
    use that definition to calculate the energy for this goal
    function: E = U^T @ U. Interestingly, this expression is
    equivalent to minimizing the squared sum of the edge lengths l
    of the network, E = l^2.

    In practice, we are interested in a single scalar value of
    this energy to minimize with optimization. As a result, we
    ravel matrix U into a single vector, and calculate the dot
    product of U in the raveled state.
    """
    @staticmethod
    def prediction(eq_state, index):
        """
        The current load path of the network.
        """
        vectors = eq_state.vectors
        laplacian = jnp.dot(jnp.ravel(vectors),
                            jnp.reshape(vectors, (-1, 1)))

        return jnp.atleast_1d(laplacian)

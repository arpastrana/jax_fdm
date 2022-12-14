import jax.numpy as jnp

from jax_fdm.constraints.network import NetworkConstraint


class NetworkEdgesLengthConstraint(NetworkConstraint):
    """
    Set constraint bounds to the length of all the edges of a network.
    """
    @staticmethod
    def constraint(eqstate, model):
        """
        The constraint function relative to a equilibrium state.
        """
        return jnp.ravel(eqstate.lengths)

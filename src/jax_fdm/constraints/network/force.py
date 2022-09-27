import jax.numpy as jnp

from jax_fdm.constraints.network import NetworkConstraint


class NetworkEdgesForceConstraint(NetworkConstraint):
    """
    Set constraint bounds to the force passing through every edge of a network.
    """
    @staticmethod
    def constraint(eqstate, model):
        """
        The constraint function relative to a equilibrium state.
        """
        return jnp.ravel(eqstate.forces)

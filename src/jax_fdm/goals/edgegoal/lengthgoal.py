from functools import partial

import jax.numpy as jnp

from jax_fdm.goals import ScalarGoal
from jax_fdm.goals.edgegoal import EdgeGoal


class EdgeLengthGoal(ScalarGoal, EdgeGoal):
    """
    Make an edge of a network to reach a certain length.
    """
    def __init__(self, key, target, weight=1.0):
        super().__init__(key=key, target=target, weight=weight)

    def prediction(self, eq_state):
        """
        The current edge length.
        """
        length = eq_state.lengths[self.index, ]
        return jnp.atleast_1d(length)

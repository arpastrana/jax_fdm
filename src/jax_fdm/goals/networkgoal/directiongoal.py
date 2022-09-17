import numpy as np
import jax.numpy as jnp

from jax_fdm.goals import ScalarGoal
from jax_fdm.goals.networkgoal import NetworkGoal


class NetworkEdgesDirectionGoal(ScalarGoal, NetworkGoal):
    """
    Make the total load path of a network to reach a target magnitude.

    The load path of an edge is the absolute value of the product of the
    the force on the edge time its length.
    """
    def __init__(self, keys=None, target=None, weight=1.0):
        super().__init__(keys=keys, target=target, weight=weight)
        self._target = np.asarray(target, dtype=np.float64)

    def prediction(self, eq_state):
        """
        The edge vectors in the network.
        """
        vector = eq_state.vectors[self.index, :]
        return vector / jnp.linalg.norm(vector, axis=1, keepdims=True)

    def target(self, prediction):
        """
        The target vectors.
        """
        return self._target / np.linalg.norm(self._target, axis=1, keepdims=True)

    def model_index(self, model):
        """
        The index of the goal key in a structure.
        """
        if self.key is None:
            edges = model.structure.network.edges()
        elif isinstance(self.key, (list, tuple)):
            edges = self.key
        return tuple([model.structure.edge_index[edge] for edge in edges])

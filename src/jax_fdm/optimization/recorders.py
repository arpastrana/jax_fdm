import numpy as np

from compas.data import Data

import jax.tree_util as jtu


# ==========================================================================
# Recorder
# ==========================================================================

class OptimizationRecorder(Data):
    """A recorder that stores data during the optimization process.
    """
    def __init__(self, optimizer=None):
        self.optimizer = optimizer
        self.history_names = ("q", "xyz_fixed", "loads")
        self.history = {name: [] for name in self.history_names}

    def __call__(self, xk, *args, **kwargs):
        if self.optimizer:
            xk = self.optimizer.parameters_fdm(xk)
        self.record(xk)

    def record(self, parameters):
        for parameter, name in zip(parameters, self.history_names):
            self.history[name].append(parameter)

    @property
    def data(self):
        data = {}
        data["history"] = jtu.tree_map(lambda leaf: np.asarray(leaf, dtype=np.float64).tolist(), self.history)
        return data

    @data.setter
    def data(self, data):
        self.history = data["history"]

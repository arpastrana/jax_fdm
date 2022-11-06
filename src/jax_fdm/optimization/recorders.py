import numpy as np

from compas.data import Data

import jax.tree_util as jtu


# ==========================================================================
# Recorder
# ==========================================================================

class OptimizationRecorder(Data):
    def __init__(self, optimizer=None):
        self.optimizer = optimizer
        self.history = []

    def record(self, value):
        self.history.append(value)

    def __call__(self, xk, *args, **kwargs):
        if self.optimizer:
            xk = self.optimizer.parameters_fdm(xk)
        self.record(xk)

    @property
    def data(self):
        data = {}
        data["history"] = jtu.tree_map(lambda leaf: np.asarray(leaf, dtype=np.float64).tolist(), self.history)
        return data

    @data.setter
    def data(self, data):
        self.history = data["history"]

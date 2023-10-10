import numpy as np

from compas.data import Data

import jax.tree_util as jtu

from jax_fdm.equilibrium import LoadState
from jax_fdm.equilibrium import EquilibriumParametersState


# ==========================================================================
# Recorder
# ==========================================================================

class OptimizationRecorder(Data):
    """A recorder that stores data during the optimization process.
    """
    def __init__(self, optimizer=None):
        self.optimizer = optimizer
        self.history = self._init_history()

    def _init_history(self):
        loads = LoadState(nodes=[], edges=[], faces=[])
        return EquilibriumParametersState(q=[], xyz_fixed=[], loads=loads)

    def __call__(self, xk, *args, **kwargs):
        if self.optimizer:
            xk = self.optimizer.parameters_fdm(xk)
        self.record(xk)

    def __len__(self):
        return len(self.history[0])

    def __getitem__(self, index):
        def index_from_leaf(leaf):
            return leaf[index]

        return jtu.tree_map(index_from_leaf,
                            self.history,
                            is_leaf=lambda x: isinstance(x, list))

    def record(self, parameters):
        def append_file(data, file):
            file.append(data)

        jtu.tree_map(append_file, parameters, self.history)

    @property
    def data(self):
        def leaf_to_list(leaf):
            return np.asarray(leaf, dtype=np.float64).tolist()

        data = {}
        history_params = jtu.tree_map(leaf_to_list,
                                      self.history,
                                      is_leaf=lambda x: isinstance(x, list))

        data["history"] = {key: val for key, val in history_params._asdict().items()}

        return data

    @data.setter
    def data(self, data):
        def leaf_to_array(leaf):
            return np.asarray(leaf, dtype=np.float64)

        # q, xyz_fixed, loads = data["history"]
        history = data["history"]
        nodes, edges, faces = history["loads"]

        loads = LoadState(nodes=nodes, edges=edges, faces=faces)
        history_params = EquilibriumParametersState(q=history["q"],
                                                    xyz_fixed=history["xyz_fixed"],
                                                    loads=loads)

        history_params = jtu.tree_map(leaf_to_array,
                                      history_params,
                                      is_leaf=lambda x: isinstance(x, list))

        self.history = history_params

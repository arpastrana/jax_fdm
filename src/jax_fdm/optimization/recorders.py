from typing import Any

import jax.tree_util as jtu
import numpy as np
from jaxtyping import Array
from jaxtyping import Float

from compas.data import Data
from jax_fdm.equilibrium import EquilibriumParametersState
from jax_fdm.equilibrium import LoadState
from jax_fdm.optimization.optimizers import Optimizer

# ==========================================================================
# Recorder
# ==========================================================================

class OptimizationRecorder(Data):
    """A recorder that stores data during the optimization process.
    """
    def __init__(self, optimizer: Optimizer | None = None):
        super().__init__()
        self.optimizer = optimizer
        self.history = self._init_history()

    def _init_history(self) -> EquilibriumParametersState | list[Any]:
        if self.optimizer:
            loads = LoadState(nodes=[], edges=[], faces=[])  # pyright: ignore[reportArgumentType]  # lists are grown in-place via record(); LoadState/EquilibriumParametersState fields are declared as Array but populated incrementally
            return EquilibriumParametersState(q=[], xyz_fixed=[], loads=loads)  # pyright: ignore[reportArgumentType]  # see above

        history: list[Any] = []
        return history

    def __call__(self, xk: Float[Array, "parameters"], *args: Any, **kwargs: Any) -> None:
        parameters: Float[Array, "parameters"] | EquilibriumParametersState = xk
        if self.optimizer:
            parameters = self.optimizer.parameters_fdm(xk)
        self.record(parameters)

    def __getitem__(self, index: int) -> Any:
        def index_from_leaf(leaf: Any) -> Any:
            return leaf[index]

        return jtu.tree_map(index_from_leaf,
                            self.history,
                            is_leaf=lambda x: isinstance(x, list))

    def __len__(self) -> int:
        if isinstance(self.history, list):
            return len(self.history)

        return len(self.history.q)

    def record(self, parameters: Any) -> None:
        def append_file(data: Any, file: Any) -> None:
            file.append(data)

        if self.optimizer:
            jtu.tree_map(append_file, parameters, self.history)
        else:
            append_file(parameters, self.history)

    @property
    def __data__(self) -> dict[str, Any]:
        def leaf_to_list(leaf: Any) -> list[Any]:
            return np.asarray(leaf, dtype=np.float64).tolist()

        data = {}
        history_params = jtu.tree_map(leaf_to_list,
                                      self.history,
                                      is_leaf=lambda x: isinstance(x, list))

        data["history"] = {key: val for key, val in history_params._asdict().items()}

        return data

    @classmethod
    def __from_data__(cls, data: dict[str, Any]) -> "OptimizationRecorder":
        def leaf_to_array(leaf: Any) -> Float[np.ndarray, "..."]:
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

        obj = cls()
        obj.history = history_params

        return obj

from os import PathLike
from typing import TYPE_CHECKING
from typing import Any
from typing import Self

import jax.tree_util as jtu
import numpy as np
from jaxtyping import Array
from jaxtyping import Float

from compas.data import Data
from jax_fdm.equilibrium import EquilibriumParametersState
from jax_fdm.equilibrium import LoadState
from jax_fdm.optimization.optimizers import Optimizer

__all__ = ["OptimizationRecorder"]

# ==========================================================================
# Recorder
# ==========================================================================


class OptimizationRecorder(Data):
    """
    A COMPAS data object that logs the parameters visited during optimization.

    Parameters
    ----------
    optimizer :
        The optimizer whose iterates are recorded. If given, each iterate is
        expanded into FDM parameters before storing; if None, raw iterates are
        stored as a flat list.

    Notes
    -----
    As a COMPAS ``Data`` subclass the history serializes to and from JSON, so an
    optimization run can be saved and replayed.
    """

    if TYPE_CHECKING:
        # Typing-only redeclaration: COMPAS types the inherited constructor
        # with a python-2 comment as `-> Data`, hiding this class's indexing
        # and history API from static checkers. Never executes; the COMPAS
        # implementation constructs through `cls` and returns this class.
        # Raising body, not `...`: pylint reads a `...` stub as returning None.
        @classmethod
        def from_json(cls, filepath: str | PathLike[str]) -> Self:
            raise NotImplementedError

    def __init__(self, optimizer: Optimizer | None = None) -> None:
        super().__init__()
        self.optimizer = optimizer
        self.history = self._init_history()

    def _init_history(self) -> EquilibriumParametersState | list[Any]:
        if self.optimizer:
            # lists are grown in-place via record(); the LoadState and
            # EquilibriumParametersState fields are declared as Array but
            # populated incrementally
            loads = LoadState(nodes=[], edges=[], faces=[])  # pyright: ignore[reportArgumentType]
            return EquilibriumParametersState(q=[], xyz_fixed=[], loads=loads)  # pyright: ignore[reportArgumentType]

        history: list[Any] = []
        return history

    def __call__(
        self,
        xk: Float[Array, "parameters"],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Record one optimizer iterate; suitable as an optimizer callback.

        Parameters
        ----------
        xk :
            The current flat optimization parameter vector.
        args :
            Extra positional callback arguments, ignored.
        kwargs :
            Extra keyword callback arguments, ignored.
        """
        parameters: Float[Array, "parameters"] | EquilibriumParametersState = xk
        if self.optimizer:
            parameters = self.optimizer.parameters_fdm(xk)
        self.record(parameters)

    def __getitem__(self, index: int) -> Any:
        def index_from_leaf(leaf: Any) -> Any:
            return leaf[index]

        return jtu.tree_map(
            index_from_leaf,
            self.history,
            is_leaf=lambda x: isinstance(x, list),
        )

    def __len__(self) -> int:
        if isinstance(self.history, list):
            return len(self.history)

        return len(self.history.q)

    def record(self, parameters: Any) -> None:
        """
        Append one set of parameters to the history.

        Parameters
        ----------
        parameters :
            The parameters to store. With an optimizer, a parameter state whose
            leaves are appended into the history's matching leaves; without one, a
            single value appended to the flat history list.
        """

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
        history_params = jtu.tree_map(
            leaf_to_list,
            self.history,
            is_leaf=lambda x: isinstance(x, list),
        )

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
        history_params = EquilibriumParametersState(
            q=history["q"],
            xyz_fixed=history["xyz_fixed"],
            loads=loads,
        )

        history_params = jtu.tree_map(
            leaf_to_array,
            history_params,
            is_leaf=lambda x: isinstance(x, list),
        )

        obj = cls()
        obj.history = history_params

        return obj

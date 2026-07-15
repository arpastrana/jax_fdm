from typing import Any

import jax.numpy as jnp
import numpy as np
from jax import vmap
from jaxtyping import Array
from jaxtyping import Float
from jaxtyping import Int

from jax_fdm import DTYPE_JAX
from jax_fdm.equilibrium import EquilibriumModel
from jax_fdm.equilibrium import EquilibriumState
from jax_fdm.equilibrium import EquilibriumStructure
from jax_fdm.goals import GoalState

# ==========================================================================
# Base goal
# ==========================================================================

class Goal:
    """
    The base goal.

    All goal subclasses must inherit from this class.
    """
    def __init__(
        self,
        key: int | tuple[int, int] | list[int] | list[tuple[int, int]],
        target: float | Float[Array, "..."],
        weight: float | Float[Array, "..."] = 1.0,
    ) -> None:
        self._key: int | tuple[int, int] | list[int] | list[tuple[int, int]] | None = None
        self._weight: Float[Array, "elements 1"]
        self._target: Float[Array, "..."]
        # set in init() from the equilibrium structure, before any prediction runs
        self._index: Int[np.ndarray, "elements"]

        self.key = key
        self.weight = weight
        self.target = target

        self.is_collectible = True

    @property
    def key(self) -> int | tuple[int, int] | list[int] | list[tuple[int, int]] | None:
        """
        The key of an element in a network.
        """
        return self._key

    @key.setter
    def key(self, key: int | tuple[int, int] | list[int] | list[tuple[int, int]]) -> None:
        # A single-goal Collection re-wraps an already-list key as [[...]] when
        # it reconstructs the goal; unwrap that extra nesting so an aggregate
        # goal (e.g. NodesColinearGoal) keeps its flat list of element keys.
        if isinstance(key, list) and len(key) == 1 and isinstance(key[0], list):
            key = key[0]
        self._key = key

    @property
    def index(self) -> Int[np.ndarray, "elements"]:
        """
        The index of the goal key in the canonical ordering of a structure.
        """
        return self._index

    @index.setter
    def index(self, index: int | tuple[int, ...] | Int[np.ndarray, "elements"]) -> None:
        if isinstance(index, int):
            index = (index,)
        self._index = np.asarray(index)

    @property
    def weight(self) -> Float[Array, "elements 1"]:
        """
        The importance of the goal.
        """
        return self._weight

    @weight.setter
    def weight(self, weight: float | Float[Array, "..."]) -> None:
        self._weight = jnp.reshape(jnp.asarray(weight, dtype=DTYPE_JAX), (-1, 1))

    @property
    def target(self) -> Float[Array, "..."]:
        """
        The target to achieve.
        """
        raise NotImplementedError

    @target.setter
    def target(self, target: float | Float[Array, "..."]) -> None:
        # Concrete goals provide the setter via the ScalarGoal / VectorGoal
        # mixins; the base only declares the contract so that Goal.__init__ can
        # assign self.target without tripping a read-only property.
        raise NotImplementedError

    def goal(self, target: Float[Array, "..."], prediction: Float[Array, "..."]) -> Float[Array, "..."]:
        """
        The goal value to compare the prediction against.
        """
        return target

    def prediction(self, eq_state: EquilibriumState, index: Int[Array, "..."]) -> Float[Array, "..."]:
        """
        The current value of the quantity of interest.
        """
        raise NotImplementedError

    def index_from_model(
        self,
        model: EquilibriumModel,
        structure: EquilibriumStructure,
    ) -> int | tuple[int, ...]:
        """
        The index of the goal key in an equilibrium structure.
        """
        raise NotImplementedError

    def _index_from_key(self, key_index: dict[Any, int]) -> int | tuple[int, ...]:
        """
        Look up the goal key in an element-to-index mapping.

        A single key maps to one index, while a list of keys maps to a tuple of indices.
        """
        if isinstance(self.key, list):
            return tuple(key_index[k] for k in self.key)
        return key_index[self.key]

    def init(self, model: EquilibriumModel, structure: EquilibriumStructure) -> None:
        """
        Initialize the goal with information from an equilibrium model.
        """
        self.index = self.index_from_model(model, structure)

    def __call__(self, eqstate: EquilibriumState) -> GoalState:
        """
        Return the current goal state.
        """
        prediction = vmap(self.prediction, in_axes=(None, 0))(eqstate, self.index)  # pyright: ignore[reportArgumentType]  # self.index is a numpy index array populated in init and mapped to a jax scalar by vmap
        goal = vmap(self.goal)(self.target, prediction)

        msg = f"Goal {self.__class__.__name__} shape: {goal.shape} vs. prediction shape: {prediction.shape}"
        assert goal.shape == prediction.shape, msg

        return GoalState(goal=goal, prediction=prediction, weight=self.weight)


# ==========================================================================
# Base goal for a scalar quantity
# ==========================================================================

class ScalarGoal:
    """
    A goal that is expressed as a scalar quantity.
    """
    @property
    def target(self) -> Float[Array, "elements 1"]:
        """
        The target to achieve.
        """
        return self._target

    @target.setter
    def target(self, target: float | Float[Array, "..."]) -> None:
        values = [target] if isinstance(target, (int, float)) else target
        self._target = jnp.reshape(jnp.asarray(values, dtype=DTYPE_JAX), (-1, 1))


# ==========================================================================
# Base goal for vector quantities
# ==========================================================================

class VectorGoal:
    """
    A goal that is expressed as a vector 3D quantity.
    """
    @property
    def target(self) -> Float[Array, "elements 3"]:
        """
        The target to achieve
        """
        return self._target

    @target.setter
    def target(self, target: Float[Array, "..."]) -> None:
        self._target = jnp.reshape(jnp.asarray(target, dtype=DTYPE_JAX), (-1, 3))

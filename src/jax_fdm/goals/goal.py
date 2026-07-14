import numpy as np
from jax import vmap
from jaxtyping import Array
from jaxtyping import Float
from jaxtyping import Int

from jax_fdm import DTYPE_NP
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
        key: int | tuple[int, int] | list,
        target: float | Float[Array, "..."] | Float[np.ndarray, "..."],
        weight: float,
    ):
        self._key = None
        self._weight = None
        self._target = None
        self._index = None

        self.key = key
        self.weight = weight
        self.target = target

        self.is_collectible = True

    @property
    def key(self):
        """
        The key of an element in a network.
        """
        return self._key

    @key.setter
    def key(self, key: int | tuple[int, int] | list) -> None:
        if isinstance(key, int) or (isinstance(key, tuple) and len(key) == 2):
            key = key
        elif sum(isinstance(i, list) for i in key) > 0:
            key = key.pop()
        self._key = key

    @property
    def index(self):
        """
        The index of the goal key in the canonical ordering of a structure.
        """
        return self._index

    @index.setter
    def index(self, index: int | Int[np.ndarray, "elements"]) -> None:
        if isinstance(index, int):
            index = [index]  # pyright: ignore[reportAssignmentType]  # reassigned to a list only to feed np.array below, not the annotated element type
        self._index = np.array(index)

    @property
    def weight(self):
        """
        The importance of the goal.
        """
        return self._weight

    @weight.setter
    def weight(self, weight: float | Float[Array, "..."] | Float[np.ndarray, "..."]) -> None:
        self._weight = np.reshape(np.asarray(weight, dtype=DTYPE_NP), (-1, 1))

    @property
    def target(self):
        """
        The target to achieve.
        """
        raise NotImplementedError

    @staticmethod
    def goal(target: Float[Array, "..."], prediction: Float[Array, "..."]) -> Float[Array, "..."]:
        """
        The target to achieve.
        """
        return target

    def init(self, model: EquilibriumModel, structure: EquilibriumStructure) -> None:
        """
        Initialize the goal with information from an equilibrium model.
        """
        self.index = self.index_from_model(model, structure)  # pyright: ignore[reportAttributeAccessIssue]  # index_from_model is defined by concrete Goal subclasses (edge/face/mesh/network/node/vertex), not on this base class

    def __call__(self, eqstate: EquilibriumState) -> GoalState:
        """
        Return the current goal state.
        """
        prediction = vmap(self.prediction, in_axes=(None, 0))(eqstate, self.index)  # pyright: ignore[reportAttributeAccessIssue]  # prediction is defined by concrete Goal subclasses, not on this base class
        goal = vmap(self.goal)(self.target, prediction)

        msg = f"Goal {self.__class__.__name__} shape: {goal.shape} vs. prediction shape: {prediction.shape}"
        assert goal.shape == prediction.shape, msg

        return GoalState(goal=goal, prediction=prediction, weight=self.weight)  # pyright: ignore[reportArgumentType]  # self.weight is Optional by declaration but always populated in __init__ before __call__ runs


# ==========================================================================
# Base goal for a scalar quantity
# ==========================================================================

class ScalarGoal:
    """
    A goal that is expressed as a scalar quantity.
    """
    @property
    def target(self):
        """
        The target to achieve.
        """
        return self._target

    @target.setter
    def target(self, target: float | Float[Array, "..."] | Float[np.ndarray, "..."]) -> None:
        if isinstance(target, (int, float)):
            target = [target]  # pyright: ignore[reportAssignmentType]  # reassigned to a list only to feed np.array below, not the annotated element type
        self._target = np.reshape(np.array(target), (-1, 1))


# ==========================================================================
# Base goal for vector quantities
# ==========================================================================

class VectorGoal:
    """
    A goal that is expressed as a vector 3D quantity.
    """
    @property
    def target(self):
        """
        The target to achieve
        """
        return self._target

    @target.setter
    def target(self, target: Float[Array, "..."] | Float[np.ndarray, "..."]) -> None:
        self._target = np.reshape(np.asarray(target, dtype=DTYPE_NP), (-1, 3))

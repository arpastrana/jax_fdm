from functools import partial

import numpy as np

from jax import jit

from jax import vmap

from jax_fdm import DTYPE_NP

from jax_fdm.goals import GoalState


# ==========================================================================
# Base goal
# ==========================================================================

class Goal:
    """
    The base goal.

    All goal subclasses must inherit from this class.
    """
    def __init__(self, key, target, weight):
        self._key = None
        self._weight = None
        self._target = None
        self._index = None

        self.key = key
        self.weight = weight
        self.target = target

    @property
    def key(self):
        """
        The key of an element in a network.
        """
        return self._key

    @key.setter
    def key(self, key):
        self._key = key

    @property
    def index(self):
        """
        The index of the goal key in the canonical ordering of a structure.
        """
        return self._index

    @index.setter
    def index(self, index):
        if isinstance(index, int):
            index = [index]
        self._index = np.array(index)

    @property
    def weight(self):
        """
        The importance of the goal.
        """
        return self._weight

    @weight.setter
    def weight(self, weight):
        self._weight = np.reshape(np.asarray(weight, dtype=DTYPE_NP), (-1, 1))

    @property
    def target(self):
        """
        The target to achieve.
        """
        raise NotImplementedError

    @staticmethod
    def goal(target, prediction):
        """
        The target to achieve.
        """
        return target

    @partial(jit, static_argnums=0)
    def __call__(self, eqstate):
        """
        Return the current goal state.
        """
        # index = self.index(model)
        prediction = vmap(self.prediction, in_axes=(None, 0))(eqstate, self.index)
        goal = vmap(self.goal)(self.target, prediction)

        assert goal.shape == prediction.shape, f"Goal shape: {goal.shape} vs. Prediction shape: {prediction.shape}"

        return GoalState(goal=goal, prediction=prediction, weight=self.weight)


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
    def target(self, target):
        if isinstance(target, (int, float)):
            target = [target]
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
    def target(self, target):
        self._target = np.reshape(np.asarray(target, dtype=DTYPE_NP), (-1, 3))

from abc import abstractmethod
from abc import abstractproperty

import numpy as np

from jax_fdm.goals import GoalState


# ==========================================================================
# Meta base goals
# ==========================================================================

class AbstractGoal:

    @abstractmethod
    def __call__(self, eqstate):
        """
        Return the current goal state.
        """
        raise NotImplementedError

    @abstractproperty
    def key(self):
        """
        The key of an element in a network.
        """
        raise NotImplementedError

    @abstractproperty
    def index(self):
        """
        The index of the goal key in the canonical ordering of a structure.
        """
        raise NotImplementedError

    @abstractproperty
    def weight(self):
        """
        The importance of the goal.
        """
        raise NotImplementedError

    @abstractmethod
    def prediction(self, eq_state):
        """
        The current reference value in the equilibrium state.
        """
        raise NotImplementedError

    @abstractmethod
    def target(self, prediction):
        """
        The target to achieve.
        """
        raise NotImplementedError

# ==========================================================================
# Base goal for a scalar quantity
# ==========================================================================


class Goal:
    """
    The base goal.

    All goal subclasses must inherit from this class.
    """
    def __init__(self, key, target, weight):
        self._key = None
        self._index = None

        self.key = key
        self._weight = weight
        self._target = target

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
        self._index = index

    def __call__(self, eqstate):
        """
        Return the current goal state.
        """
        prediction = self.prediction(eqstate)
        target = self.target(prediction)
        weight = self.weight()

        return GoalState(target=target, prediction=prediction, weight=weight)

# ==========================================================================
# Base goal for a scalar quantity
# ==========================================================================


class ScalarGoal:
    """
    A goal that is expressed as a scalar quantity.
    """
    def weight(self):
        """
        The importance of the goal
        """
        return np.array(self._weight)

    def target(self, prediction):
        """
        The target to strive for.
        """
        return np.array(self._target)

# ==========================================================================
# Base goal for vector quantities
# ==========================================================================


class VectorGoal:
    """
    A goal that is expressed as a vector 3D quantity.
    """
    def weight(self):
        """
        The importance of the goal
        """
        return np.asarray(self._weight, dtype=np.float64)

    def target(self, prediction):
        """
        The target to strive for.
        """
        return np.asarray(self._target, dtype=np.float64)

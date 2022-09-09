from abc import abstractmethod

import jax.numpy as jnp

from jax_fdm.goals import GoalState


# ==========================================================================
# Meta base goals
# ==========================================================================

class Goal:
    """
    The base goal.

    All goal subclasses must inherit from this class.
    """
    def __init__(self, key, target, weight):
        self._key = key
        self._target = target
        self._weight = weight

    def __call__(self, eqstate, model):
        """
        Return the current goal state.
        """
        prediction = self.prediction(eqstate, self.index(model))
        target = self.target(prediction)
        weight = self.weight()

        return GoalState(target=target,
                         prediction=prediction,
                         weight=weight)

    @abstractmethod
    def key(self):
        """
        The key of an element in a network.
        """
        raise NotImplementedError

    @abstractmethod
    def weight(self):
        """
        The importance of the goal.
        """
        raise NotImplementedError

    @abstractmethod
    def prediction(self, eq_state, index):
        """
        The current reference value in the equilibrium state.
        """
        raise NotImplementedError

    @abstractmethod
    def target(self, prediction):
        """
        The target to strive for.
        """
        raise NotImplementedError

    @abstractmethod
    def index(self, structure):
        """
        The index of the goal key in the canonical ordering of the equilibrium structure.
        """
        raise NotImplementedError


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
        return jnp.atleast_1d(self._weight)

    def target(self, prediction):
        """
        The target to strive for.
        """
        return jnp.atleast_1d(self._target)

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
        return jnp.asarray([self._weight] * 3, dtype=jnp.float64)

    def target(self, prediction):
        """
        The target to strive for.
        """
        return jnp.asarray(self._target, dtype=jnp.float64)

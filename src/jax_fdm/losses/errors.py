from functools import partial

import jax

import jax.numpy as jnp

from jax import jit


# ==========================================================================
# Error
# ==========================================================================

class Error:
    def __init__(self, goals, alpha=1.0, name=None, *args, **kwargs):
        self.goals = goals
        self.alpha = alpha
        self.name = name or self.__class__.__name__

    @staticmethod
    def error(errors):
        raise NotImplementedError

    @staticmethod
    def errors(gstate):
        raise NotImplementedError

    @partial(jit, static_argnums=0)
    def __call__(self, eqstate):
        func = partial(self._error_goal, eqstate=eqstate)
        errors = jax.tree_map(func, self.goals)
        return self.errors(jnp.array(errors)) * self.alpha

    def _error_goal(self, goal, eqstate):
        return self.error(goal(eqstate))

# ==========================================================================
# Precooked error functions
# ==========================================================================


class SquaredError(Error):
    """
    The canonical squared error.
    Measures the distance between the current and the target value of a goal.
    """
    @staticmethod
    def error(gstate):
        return jnp.sum(gstate.weight * jnp.square(gstate.prediction - gstate.target))

    @staticmethod
    def errors(errors):
        return jnp.sum(errors)


class MeanSquaredError(SquaredError):
    """
    The seminal mean squared error.
    Average out all errors because no single error is important enough.
    """
    @staticmethod
    def error(errors):
        return jnp.mean(errors)


class PredictionError(Error):
    """
    You lose when you predict too much of something.
    """
    @staticmethod
    def error(gstate):
        return gstate.prediction

    @staticmethod
    def errors(errors):
        return jnp.sum(errors)

from functools import partial

import jax.numpy as jnp

from jax import jit

from jax_fdm.goals import goals_state


# ==========================================================================
# Loss
# ==========================================================================

class LossTerm:
    def __init__(self, goals, alpha=1.0, name=None, *args, **kwargs):
        self.goals = goals
        self.alpha = alpha
        self._name = name

    @property
    def name(self):
        if not self._name:
            self._name = self.__class__.__name__
        return self._name

    @partial(jit, static_argnums=(0, 2))
    def __call__(self, eqstate, model):
        gstate = self.goals_state(eqstate, model)
        return self.alpha * self.loss(gstate)

    @partial(jit, static_argnums=(0, 2))
    def goals_state(self, eqstate, model):
        return goals_state(self.goals, eqstate, model)

    @staticmethod
    def loss(gstate):
        raise NotImplementedError

# ==========================================================================
# Precooked loss functions
# ==========================================================================


class SquaredError(LossTerm):
    """
    The canonical squared error.
    Measures the distance between the current and the target value of a goal.
    """
    @staticmethod
    @jit
    def loss(gstate):
        return jnp.sum(gstate.weight * jnp.square(gstate.prediction - gstate.target))


class MeanSquaredError(LossTerm):
    """
    The seminal mean squared error.
    Average out all errors because no single error is important enough.
    """
    @staticmethod
    @jit
    def loss(gstate):
        return jnp.mean(gstate.weight * jnp.square(gstate.prediction - gstate.target))


class PredictionError(LossTerm):
    """
    You lose when you predict too much of something.
    """
    @staticmethod
    @jit
    def loss(gstate):
        return jnp.sum(gstate.prediction)

# ==========================================================================
# Loss
# ==========================================================================


class Loss:
    def __init__(self, *args, name=None):
        self.loss_terms = args
        self._name = name

    @partial(jit, static_argnums=(0, 2))
    def __call__(self, q, model):
        eqstate = model(q)
        error = 0.0
        for loss in self.loss_terms:
            error = error + loss(eqstate, model)
        return error

    @property
    def name(self):
        if not self._name:
            self._name = self.__class__.__name__
        return self._name

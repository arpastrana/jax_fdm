import jax.numpy as jnp

from dfdm.goals import goals_state


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

    def __call__(self, eqstate, model):
        gstate = self.goals_state(eqstate, model)
        return self.alpha * self.loss(gstate)

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
    def loss(gstate):
        return jnp.sum(gstate.weight * jnp.square(gstate.prediction - gstate.target))


class MeanSquaredError(LossTerm):
    """
    The seminal mean squared error loss.
    Average out all errors because no single error is important enough.
    """
    @staticmethod
    def loss(gstate):
        return jnp.mean(gstate.weight * jnp.square(gstate.prediction - gstate.target))


class PredictionError(LossTerm):
    """
    You lose when you predict too much of something.
    """
    @staticmethod
    def loss(gstate):
        return jnp.sum(gstate.prediction)

# ==========================================================================
# Loss
# ==========================================================================


class Loss:
    def __init__(self, *args, name=None):
        self.loss_terms = args
        self._name = name

    def __call__(self, q, model):
        eqstate = model(q)
        error = 0.0
        for loss in self.loss_terms:
            error += loss(eqstate, model)
        return error

    @property
    def name(self):
        if not self._name:
            self._name = self.__class__.__name__
        return self._name

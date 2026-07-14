import jax.numpy as jnp
from jaxtyping import Array
from jaxtyping import Float

from jax_fdm.equilibrium import EquilibriumState
from jax_fdm.goals import Goal
from jax_fdm.goals import GoalState

# ==========================================================================
# Error
# ==========================================================================

class Error:
    """
    The base class for an error term in a loss function.
    """
    def __init__(self, goals: list[Goal], alpha: float = 1.0, name: str | None = None, *args, **kwargs):
        self.goals = goals
        self.alpha = alpha
        self.name = name or self.__class__.__name__
        self.collections = []

    @staticmethod
    def error(gstate: GoalState) -> Float[Array, ""]:
        """
        The value of the error term.
        """
        raise NotImplementedError

    def errors(self, errors: Float[Array, "errors"]) -> Float[Array, ""]:
        """
        The sum of the individual error terms.
        """
        return jnp.sum(errors)

    def __call__(self, eqstate: EquilibriumState) -> Float[Array, ""]:
        """
        Return the current value of the error term.
        """
        errors = []
        for goal_collection in self.collections:
            gstate = goal_collection(eqstate)
            error = self.error(gstate)
            errors.append(error)

        return self.errors(jnp.array(errors)) * self.alpha

    def number_of_goals(self) -> int:
        """
        The total number of individual goals in this error term.
        """
        return len(self.goals)

    def number_of_collections(self) -> int:
        """
        The total number of goal collections in this error term.
        """
        return len(self.collections)


# ==========================================================================
# Precooked error terms
# ==========================================================================

class SquaredError(Error):
    """
    The canonical squared error.

    It measures the L2 distance between the current and the target value of a goal.
    """
    @staticmethod
    def error(gstate: GoalState) -> Float[Array, ""]:
        return jnp.sum(gstate.weight * jnp.square(gstate.prediction - gstate.goal))


class MeanSquaredError(SquaredError):
    """
    The seminal mean squared error.

    Average out all errors because no single error is important enough.
    """
    def errors(self, errors: Float[Array, "errors"]) -> Float[Array, ""]:
        """
        The mean of the individual error terms.
        """
        return super(MeanSquaredError, self).errors(errors) / self.number_of_goals()


class RootMeanSquaredError(MeanSquaredError):
    """
    The root mean squared error.
    """
    def errors(self, errors: Float[Array, "errors"]) -> Float[Array, ""]:
        """
        The root of the mean of the individual error terms.
        """
        error = super(RootMeanSquaredError, self).errors(errors)
        return jnp.sqrt(error)


class PredictionError(Error):
    """
    The prediction error.

    You lose when you get too much of something.
    """
    @staticmethod
    def error(gstate: GoalState) -> Float[Array, ""]:
        """
        The value of the prediction error.
        """
        return jnp.sum(gstate.prediction * gstate.weight)


class MeanPredictionError(PredictionError):
    """
    The mean prediction error.

    Average out all errors because no single error is important enough.
    """
    def errors(self, errors: Float[Array, "errors"]) -> Float[Array, ""]:
        """
        The mean of the individual prediction error terms.
        """
        return super().errors(errors) / self.number_of_goals()


class AbsoluteError(Error):
    """
    The canonical absolute error.

    It measures the absolute difference between the current and the target value of a goal.
    """
    @staticmethod
    def error(gstate: GoalState) -> Float[Array, ""]:
        """
        The value of the absolute error.
        """
        return jnp.sum(gstate.weight * jnp.abs(gstate.prediction - gstate.goal))


class MeanAbsoluteError(AbsoluteError):
    """
    The canonical mean absolute error.
    """
    def errors(self, errors: Float[Array, "errors"]) -> Float[Array, ""]:
        """
        The mean of the individual absolute error terms.
        """
        return super().errors(errors) / self.number_of_goals()


class LogMaxError(Error):
    """
    The log error for constraints with a target maximum value that should not be exceeded.

    Helpful to deal with soft barrier constraints with an upper bound.
    """
    @staticmethod
    def error(gstate: GoalState) -> Float[Array, ""]:
        """
        The log error.
        """
        difference = gstate.prediction - gstate.goal
        # TODO: consider softplus as a smooth alternative
        violation = jnp.maximum(difference, 0.0)
        # NOTE: shifting difference via (x + 1.0) so that error is log(1) = 0.0 at least.
        # jnp.log1p(x) is equivalent but more numerically stable than jnp.log(x + 1.0)

        # TODO: consider changing to quadratic error for large violations later
        error = jnp.log1p(violation)

        return jnp.sum(gstate.weight * error)

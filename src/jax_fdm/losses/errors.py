from collections.abc import Sequence

import jax.numpy as jnp
from jaxtyping import Array
from jaxtyping import Float

from jax_fdm.equilibrium import EquilibriumState
from jax_fdm.goals import Goal
from jax_fdm.goals import GoalState

# ==========================================================================
# Error
# ==========================================================================

__all__ = [
    "AbsoluteError",
    "Error",
    "LogMaxError",
    "MeanAbsoluteError",
    "MeanPredictionError",
    "MeanSquaredError",
    "PredictionError",
    "RootMeanSquaredError",
    "SquaredError",
]


class Error:
    """
    The base class for an error term in a loss function.

    Parameters
    ----------
    goals :
        The goals whose gap from their target this term penalizes.
    alpha :
        A scalar weight scaling the whole term in the loss.
    name :
        The name of the error term. If None, defaults to the class name.

    Notes
    -----
    Goals are grouped into collections in ``collections``; the term evaluates one
    error per collection and aggregates them via `errors`.
    """

    def __init__(
        self,
        goals: Sequence[Goal],
        alpha: float = 1.0,
        name: str | None = None,
        *args,
        **kwargs,
    ) -> None:
        self.goals = goals
        self.alpha = alpha
        self.name = name or self.__class__.__name__
        self.collections: list[Goal] = []

    @staticmethod
    def error(gstate: GoalState) -> Float[Array, ""]:
        """
        The error of a single goal collection.

        Parameters
        ----------
        gstate :
            The evaluated goal state to measure.

        Returns
        -------
        error :
            The error for the collection.
        """
        raise NotImplementedError

    def errors(self, errors: Float[Array, "errors"]) -> Float[Array, ""]:
        """
        Aggregate the per-collection errors into a scalar.

        Parameters
        ----------
        errors :
            The error of each goal collection.

        Returns
        -------
        error :
            The aggregate error; the base term returns the sum.
        """
        return jnp.sum(errors)

    def __call__(self, eqstate: EquilibriumState) -> Float[Array, ""]:
        """
        Evaluate the error term against an equilibrium state.

        Parameters
        ----------
        eqstate :
            The equilibrium state to evaluate the goals on.

        Returns
        -------
        error :
            The aggregated error scaled by ``alpha``.
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

    It measures the weighted squared L2 gap between each goal's prediction
    and target.
    """

    @staticmethod
    def error(gstate: GoalState) -> Float[Array, ""]:
        """
        The weighted sum of squared prediction-target gaps in a collection.

        Parameters
        ----------
        gstate :
            The evaluated goal state to measure.

        Returns
        -------
        error :
            The weighted squared error.
        """
        return jnp.sum(gstate.weight * jnp.square(gstate.prediction - gstate.goal))


class MeanSquaredError(SquaredError):
    """
    The seminal mean squared error.

    Average out all errors because no single error is important enough.
    """

    def errors(self, errors: Float[Array, "errors"]) -> Float[Array, ""]:
        """
        The mean of the per-collection squared errors.

        Parameters
        ----------
        errors :
            The error of each goal collection.

        Returns
        -------
        error :
            The summed error divided by the number of goals.
        """
        return super(MeanSquaredError, self).errors(errors) / self.number_of_goals()


class RootMeanSquaredError(MeanSquaredError):
    """
    The square root of the mean squared error.
    """

    def errors(self, errors: Float[Array, "errors"]) -> Float[Array, ""]:
        """
        The root of the mean squared error.

        Parameters
        ----------
        errors :
            The error of each goal collection.

        Returns
        -------
        error :
            The square root of the mean squared error.
        """
        error = super(RootMeanSquaredError, self).errors(errors)
        return jnp.sqrt(error)


class PredictionError(Error):
    """
    The prediction error.

    You lose when you get too much of something.

    Notes
    -----
    Penalizes the magnitude of the predicted quantity directly rather than its gap
    from a target, useful for minimizing quantities such as load path.
    """

    @staticmethod
    def error(gstate: GoalState) -> Float[Array, ""]:
        """
        The weighted sum of the predictions in a collection.

        Parameters
        ----------
        gstate :
            The evaluated goal state to measure.

        Returns
        -------
        error :
            The weighted sum of the predictions.
        """
        return jnp.sum(gstate.prediction * gstate.weight)


class MeanPredictionError(PredictionError):
    """
    The mean prediction error.

    Average out all errors because no single error is important enough.
    """

    def errors(self, errors: Float[Array, "errors"]) -> Float[Array, ""]:
        """
        The mean of the per-collection prediction errors.

        Parameters
        ----------
        errors :
            The error of each goal collection.

        Returns
        -------
        error :
            The summed error divided by the number of goals.
        """
        return super().errors(errors) / self.number_of_goals()


class AbsoluteError(Error):
    """
    The canonical absolute error.

    It measures the weighted absolute gap between each goal's prediction
    and target.
    """

    @staticmethod
    def error(gstate: GoalState) -> Float[Array, ""]:
        """
        The weighted sum of absolute prediction-target gaps in a collection.

        Parameters
        ----------
        gstate :
            The evaluated goal state to measure.

        Returns
        -------
        error :
            The weighted absolute error.
        """
        return jnp.sum(gstate.weight * jnp.abs(gstate.prediction - gstate.goal))


class MeanAbsoluteError(AbsoluteError):
    """
    The canonical mean absolute error.
    """

    def errors(self, errors: Float[Array, "errors"]) -> Float[Array, ""]:
        """
        The mean of the per-collection absolute errors.

        Parameters
        ----------
        errors :
            The error of each goal collection.

        Returns
        -------
        error :
            The summed error divided by the number of goals.
        """
        return super().errors(errors) / self.number_of_goals()


class LogMaxError(Error):
    """
    A soft one-sided barrier penalizing goal predictions above their target.

    Notes
    -----
    Only positive overshoots past the target are penalized, through ``log1p`` of
    the violation, so the error is zero while the prediction stays at or below the
    target. Useful for soft upper-bound constraints.
    """

    @staticmethod
    def error(gstate: GoalState) -> Float[Array, ""]:
        """
        The weighted log-barrier penalty on target overshoot in a collection.

        Parameters
        ----------
        gstate :
            The evaluated goal state to measure.

        Returns
        -------
        error :
            The weighted ``log1p`` penalty on the positive prediction-target gap.
        """
        difference = gstate.prediction - gstate.goal
        # TODO: consider softplus as a smooth alternative
        violation = jnp.maximum(difference, 0.0)
        # NOTE: shifting difference via (x + 1.0) so that error is log(1) = 0.0 at
        # least.
        # jnp.log1p(x) is equivalent but more numerically stable than jnp.log(x + 1.0)

        # TODO: consider changing to quadratic error for large violations later
        error = jnp.log1p(violation)

        return jnp.sum(gstate.weight * error)

from collections.abc import Sequence

import jax.numpy as jnp
from jax import vmap
from jaxtyping import Array
from jaxtyping import Float

from jax_fdm.datastructures import FDMesh
from jax_fdm.datastructures import FDNetwork
from jax_fdm.equilibrium import EquilibriumState
from jax_fdm.equilibrium import EquilibriumStructure
from jax_fdm.equilibrium import datastructure_state
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
    Goals are grouped into collections in ``collections``: one collection per goal
    type, its leaves carrying a leading element axis. A goal maps one element to
    one error, so the term `vmap`s that per-element error over a collection, sums
    the mapped result, and aggregates one such sum per collection via `errors`.
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

    def __call__(
        self,
        eqstate: EquilibriumState,
        structure: EquilibriumStructure,
    ) -> Float[Array, ""]:
        """
        Evaluate the error term against an equilibrium state.

        Parameters
        ----------
        eqstate :
            The equilibrium state to evaluate the goals on.
        structure :
            The structure whose element ordering resolves the goals' indices.

        Returns
        -------
        error :
            The aggregated error scaled by ``alpha``.

        Notes
        -----
        A goal maps one element to one goal state, so each collection is reduced
        the equinox way: `vmap` the per-element error over the collection's leading
        axis and sum the mapped scalars, exactly as one maps any module over a
        batch. The weight, one scalar per element, broadcasts against the
        prediction at each element, so no rank alignment is needed. This is the
        same per-element reduction as `evaluate_state`, over the batched
        collections rather than the raw goals.
        """
        errors = []
        for collection in self.collections:
            per_element = vmap(lambda g: self.error(g(eqstate, structure)))(collection)
            errors.append(jnp.sum(per_element))

        return self.errors(jnp.asarray(errors)) * self.alpha

    def evaluate(
        self,
        datastructure: FDNetwork | FDMesh,
        sparse: bool = True,
    ) -> Float[Array, ""]:
        """
        Evaluate the error term directly on a datastructure, without an optimization.

        Parameters
        ----------
        datastructure :
            The network or mesh to read the equilibrium state from. Its geometry
            is used as-is; no form-finding is run.
        sparse :
            If True, assemble the equilibrium state with the sparse model.

        Returns
        -------
        error :
            The aggregated error scaled by ``alpha``.

        Notes
        -----
        Evaluates the term's raw goals one by one, rather than the collections
        the optimizer batches them into, so it works before ``constrained_fdm``
        has grouped them. The goal count is unchanged, so a mean-style term
        divides by the same number of goals it would during an optimization.
        """
        equilibrium = datastructure_state(datastructure, sparse)

        return self.evaluate_state(equilibrium.eq_state, equilibrium.structure)

    def evaluate_state(
        self,
        eqstate: EquilibriumState,
        structure: EquilibriumStructure,
    ) -> Float[Array, ""]:
        """
        Evaluate the error term's raw goals on a precomputed equilibrium state.

        Parameters
        ----------
        eqstate :
            The equilibrium state to evaluate the goals on.
        structure :
            The structure whose element ordering resolves the goals' indices.

        Returns
        -------
        error :
            The aggregated error scaled by ``alpha``.

        Notes
        -----
        The state-consuming core shared by `evaluate` and `Loss.evaluate`. It
        evaluates the term's raw goals as singletons rather than the collections
        the optimizer batches them into. A lone goal's `__call__` already returns
        an unbatched state whose scalar weight broadcasts against the prediction,
        and the error kernels sum over every axis, so a raw goal state feeds them
        directly without the collection's `(elements, features)` formatting.
        """
        errors = [self.error(goal(eqstate, structure)) for goal in self.goals]

        return self.errors(jnp.asarray(errors)) * self.alpha

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

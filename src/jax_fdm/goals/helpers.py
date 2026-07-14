import numpy as np

from jax_fdm.equilibrium import EquilibriumModel
from jax_fdm.equilibrium import EquilibriumState
from jax_fdm.goals.goal import Goal
from jax_fdm.goals.state import GoalState

# ==========================================================================
# Goal manager
# ==========================================================================

def goals_reindex(goals: list[Goal], model: EquilibriumModel) -> None:
    """
    Compute the index of a goal based on its node or edge key.
    """
    for goal in goals:
        index = goal.index_from_model(model, goal.key)  # pyright: ignore[reportAttributeAccessIssue]  # index_from_model is defined by concrete Goal subclasses, not on the base class; this helper is unused elsewhere in the codebase
        goal.index = index


def goals_state(goals: list[Goal], eqstate: EquilibriumState, model: EquilibriumModel) -> GoalState:
    """
    Collate goals attributes into vectors.
    """
    predictions = []
    targets = []
    weights = []

    for goal in goals:
        gstate = goal(eqstate, model)  # pyright: ignore[reportCallIssue]  # Goal.__call__ takes only eqstate; this helper is unused elsewhere and predates that signature
        predictions.append(gstate.prediction)
        targets.append(gstate.target)  # pyright: ignore[reportAttributeAccessIssue]  # GoalState has no `target` field (only goal/weight/prediction); this helper is unused elsewhere and predates the current GoalState shape
        weights.append(gstate.weight)

    predictions = np.concatenate(predictions, axis=0)
    targets = np.concatenate(targets, axis=0)
    weights = np.concatenate(weights, axis=0)

    return GoalState(prediction=predictions, target=targets, weight=weights)  # pyright: ignore[reportCallIssue]  # GoalState has no `target` field; this helper is unused elsewhere in the codebase and predates the current GoalState shape

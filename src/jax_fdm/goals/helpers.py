import numpy as np

import jax.numpy as jnp

from jax_fdm.goals import GoalState


# ==========================================================================
# Goal manager
# ==========================================================================

def goals_reindex(goals, model):
    """
    Compute the index of a goal based on its node or edge key.
    """
    for goal in goals:
        index = goal.model_index(model)
        goal.index = index


def goals_state(goals, eqstate, model):
    """
    Collate goals attributes into vectors.
    """
    predictions = []
    targets = []
    weights = []

    for goal in goals:
        gstate = goal(eqstate, model)
        predictions.append(gstate.prediction)
        targets.append(gstate.target)
        weights.append(gstate.weight)

    predictions = np.concatenate(predictions, axis=0)
    targets = np.concatenate(targets, axis=0)
    weights = np.concatenate(weights, axis=0)

    return GoalState(prediction=predictions, target=targets, weight=weights)


"""
q = np.random(n)
eqstate = fdm(q)

loss = 0.0
for goal in goals:
    gstate = goal(eqstate)
    loss += miniloss(gstate)
return loss
"""

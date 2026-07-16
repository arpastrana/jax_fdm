from typing import NamedTuple

from jaxtyping import Array
from jaxtyping import Float


class GoalState(NamedTuple):
    """
    The evaluated state of a goal at an equilibrium state.

    Attributes
    ----------
    goal :
        The reference value the prediction is compared against.
    weight :
        The relative importance of the goal in the loss.
    prediction :
        The current value of the goal's quantity of interest.
    """

    goal: Float[Array, "..."]
    weight: Float[Array, "..."]
    prediction: Float[Array, "..."]

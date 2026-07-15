from typing import NamedTuple

from jaxtyping import Array
from jaxtyping import Float


class GoalState(NamedTuple):
    goal: Float[Array, "..."]
    weight: Float[Array, "..."]
    prediction: Float[Array, "..."]

from typing import NamedTuple

import numpy as np
from jaxtyping import Array
from jaxtyping import Float


class GoalState(NamedTuple):
    goal: Float[np.ndarray, "..."]
    weight: Float[np.ndarray, "..."]
    prediction: Float[Array, "..."]

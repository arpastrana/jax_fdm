from typing import NamedTuple

import numpy as np
import jax.numpy as jnp


class GoalState(NamedTuple):
    goal: np.ndarray
    weight: np.ndarray
    prediction: jnp.ndarray

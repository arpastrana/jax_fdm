from typing import NamedTuple

import jax.numpy as jnp
import numpy as np


class GoalState(NamedTuple):
    goal: np.ndarray
    weight: np.ndarray
    prediction: jnp.ndarray

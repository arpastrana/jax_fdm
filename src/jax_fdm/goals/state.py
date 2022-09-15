from typing import NamedTuple

import jax.numpy as jnp


class GoalState(NamedTuple):
    prediction: jnp.ndarray
    target: jnp.ndarray
    weight: jnp.ndarray

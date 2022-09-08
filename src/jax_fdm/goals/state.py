from dataclasses import dataclass

import jax.numpy as jnp


@dataclass
class GoalState:
    prediction: jnp.ndarray
    target: jnp.ndarray
    weight: jnp.ndarray

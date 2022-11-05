from typing import NamedTuple

import jax.numpy as jnp


# ==========================================================================
# Equilibrium state
# ==========================================================================

class EquilibriumState(NamedTuple):
    xyz: jnp.ndarray
    residuals: jnp.ndarray
    lengths: jnp.ndarray
    forces: jnp.ndarray
    force_densities: jnp.ndarray
    vectors: jnp.ndarray
    loads: jnp.ndarray

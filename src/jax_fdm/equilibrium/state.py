from dataclasses import dataclass

import jax.numpy as jnp


# ==========================================================================
# Equilibrium state
# ==========================================================================

# TODO: A method that reindexes state arrays to match network indexing
@dataclass
class EquilibriumState:
    xyz: jnp.ndarray
    residuals: jnp.ndarray
    lengths: jnp.ndarray
    forces: jnp.ndarray
    force_densities: jnp.ndarray

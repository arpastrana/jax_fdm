from typing import NamedTuple
from typing import Union

import jax
import jax.numpy as jnp
from jax.typing import DTypeLike

from jax_fdm import DTYPE_JAX
from jax_fdm.datastructures import FDMesh
from jax_fdm.datastructures import FDNetwork

# ==========================================================================
# Equilibrium state
# ==========================================================================

class EquilibriumState(NamedTuple):
    xyz: jax.Array
    residuals: jax.Array
    lengths: jax.Array
    forces: jax.Array
    loads: jax.Array
    vectors: jax.Array


# ==========================================================================
# Load state
# ==========================================================================

class LoadState(NamedTuple):
    nodes: Union[jax.Array]  # pyright: ignore[reportInvalidTypeArguments]  # SMELL: single-arg Union[jax.Array] is vestigial, equivalent to plain jax.Array; left as-is per scope
    edges: Union[jax.Array, float]
    faces: Union[jax.Array, float]

    @classmethod
    def from_datastructure(cls, datastructure: FDNetwork | FDMesh, dtype: DTypeLike | None = None) -> "LoadState":
        """
        Create a load state from a datastructure.
        """
        loads_edges = jnp.asarray(datastructure.edges_loads(), dtype)
        if jnp.allclose(loads_edges, 0.0):
            loads_edges = 0.0

        if isinstance(datastructure, FDNetwork):
            loads_nodes = jnp.asarray(datastructure.nodes_loads())
            loads_faces = 0.0

        elif isinstance(datastructure, FDMesh):
            loads_nodes = jnp.asarray(datastructure.vertices_loads(), dtype)

            loads_faces = jnp.asarray(datastructure.faces_loads(), dtype)
            if jnp.allclose(loads_faces, 0.0):
                loads_faces = 0.0

        return cls(nodes=loads_nodes,
                   edges=loads_edges,
                   faces=loads_faces)


# ==========================================================================
# Parameter state
# ==========================================================================

class EquilibriumParametersState(NamedTuple):
    q: jax.Array
    xyz_fixed: jax.Array
    loads: LoadState

    @classmethod
    def from_datastructure(cls, datastructure, dtype=None):
        """
        Create a parameter state from a datastructure.
        """
        q = datastructure.edges_forcedensities()

        if isinstance(datastructure, FDNetwork):
            xyz_fixed = datastructure.nodes_fixedcoordinates()

        elif isinstance(datastructure, FDMesh):
            xyz_fixed = datastructure.vertices_fixedcoordinates()

        if dtype is None:
            dtype = DTYPE_JAX

        return cls(q=jnp.asarray(q, dtype),
                   xyz_fixed=jnp.asarray(xyz_fixed, dtype),  # pyright: ignore[reportPossiblyUnboundVariable]  # SMELL: xyz_fixed unbound if datastructure is neither FDNetwork nor FDMesh (no else branch); cannot occur in practice, left as-is per scope
                   loads=LoadState.from_datastructure(datastructure))

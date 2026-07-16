from typing import NamedTuple

import jax.numpy as jnp
from jax.typing import DTypeLike
from jaxtyping import Array
from jaxtyping import Float

from jax_fdm import DTYPE_JAX
from jax_fdm.datastructures import FDMesh
from jax_fdm.datastructures import FDNetwork

# ==========================================================================
# Equilibrium state
# ==========================================================================


class EquilibriumState(NamedTuple):
    xyz: Float[Array, "nodes 3"]
    residuals: Float[Array, "nodes 3"]
    lengths: Float[Array, "edges 1"]
    forces: Float[Array, "edges 1"]
    loads: Float[Array, "nodes 3"]
    vectors: Float[Array, "edges 3"]


# ==========================================================================
# Load state
# ==========================================================================


class LoadState(NamedTuple):
    nodes: Float[Array, "nodes 3"]
    edges: Float[Array, "edges 3"] | float
    faces: Float[Array, "faces 3"] | float

    @classmethod
    def from_datastructure(
        cls,
        datastructure: FDNetwork | FDMesh,
        dtype: DTypeLike | None = None,
    ) -> "LoadState":
        """
        Create a load state from a datastructure.
        """
        loads_edges = jnp.asarray(datastructure.edges_loads(), dtype)
        if jnp.allclose(loads_edges, 0.0):
            loads_edges = 0.0

        if isinstance(datastructure, FDNetwork):
            loads_nodes = jnp.asarray(datastructure.nodes_loads(), dtype)
            loads_faces = 0.0

        elif isinstance(datastructure, FDMesh):
            loads_nodes = jnp.asarray(datastructure.vertices_loads(), dtype)

            loads_faces = jnp.asarray(datastructure.faces_loads(), dtype)
            if jnp.allclose(loads_faces, 0.0):
                loads_faces = 0.0

        return cls(nodes=loads_nodes, edges=loads_edges, faces=loads_faces)


# ==========================================================================
# Parameter state
# ==========================================================================


class EquilibriumParametersState(NamedTuple):
    q: Float[Array, "edges"]
    xyz_fixed: Float[Array, "nodes_fixed 3"]
    loads: LoadState

    @classmethod
    def from_datastructure(
        cls,
        datastructure: FDNetwork | FDMesh,
        dtype: DTypeLike | None = None,
    ) -> "EquilibriumParametersState":
        """
        Create a parameter state from a datastructure.
        """
        q = datastructure.edges_forcedensities()

        if isinstance(datastructure, FDNetwork):
            xyz_fixed = datastructure.nodes_fixedcoordinates()
        elif isinstance(datastructure, FDMesh):
            xyz_fixed = datastructure.vertices_fixedcoordinates()
        else:
            raise ValueError(f"Input datastructure {datastructure} is invalid")

        if dtype is None:
            dtype = DTYPE_JAX

        return cls(
            q=jnp.asarray(q, dtype),
            xyz_fixed=jnp.asarray(xyz_fixed, dtype),
            loads=LoadState.from_datastructure(datastructure),
        )

from functools import partial

import numpy as np

import jax.numpy as jnp

from jax_fdm.equilibrium.state import EquilibriumState
from jax_fdm.equilibrium.structure import EquilibriumStructure

from jax import jit

# ==========================================================================
# Equilibrium model
# ==========================================================================


class EquilibriumModel:
    """
    The equilibrium solver.
    """
    def __init__(self, network):
        self.structure = EquilibriumStructure(network)
        self.loads = np.asarray(list(network.nodes_loads()), dtype=np.float64)
        self.xyz_fixed = np.asarray([network.node_coordinates(node) for node in network.nodes_fixed()], dtype=np.float64)

    @partial(jit, static_argnums=0)
    def _edges_vectors(self, xyz):
        return self.structure.connectivity @ xyz

    @partial(jit, static_argnums=0)
    def _edges_lengths(self, vectors):
        return jnp.linalg.norm(vectors, axis=1, keepdims=True)

    @partial(jit, static_argnums=0)
    def _edges_forces(self, q, lengths):
        return jnp.reshape(q, (-1, 1)) * lengths

    @partial(jit, static_argnums=0)
    def _nodes_residuals(self, q, xyz, vectors):
        connectivity = self.structure.connectivity
        return self.loads - np.transpose(connectivity) @ jnp.diag(q) @ vectors

    @partial(jit, static_argnums=0)
    def _nodes_free_positions(self, q):
        # convenience shorthands
        free = self.structure.free_nodes
        loads = self.loads
        xyz_fixed = self.xyz_fixed

        # connectivity
        c_fixed = self.structure.connectivity_fixed
        c_free = self.structure.connectivity_free
        c_free_t = np.transpose(c_free)

        # Mutable stuff
        q_matrix = jnp.diag(q)

        # solve equilibrium after solving a linear system of equations
        A = c_free_t @ q_matrix @ c_free
        b = loads[free, :] - c_free_t @ q_matrix @ c_fixed @ xyz_fixed

        return jnp.linalg.solve(A, b)

    @partial(jit, static_argnums=0)
    def _nodes_positions(self, xyz_free):
        # NOTE: free fixed indices sorted by enumeration
        xyz_fixed = self.xyz_fixed
        indices = self.structure.freefixed_nodes
        return jnp.concatenate((xyz_free, xyz_fixed))[indices, :]

    @partial(jit, static_argnums=0)
    def __call__(self, q):
        """
        Compute an equilibrium state using the force density method.
        """
        xyz_free = self._nodes_free_positions(q)
        xyz = self._nodes_positions(xyz_free)
        vectors = self._edges_vectors(xyz)
        residuals = self._nodes_residuals(q, xyz, vectors)
        lengths = self._edges_lengths(vectors)
        forces = self._edges_forces(q, lengths)

        return EquilibriumState(xyz=xyz,
                                residuals=residuals,
                                lengths=lengths,
                                forces=forces,
                                vectors=vectors,
                                force_densities=q)

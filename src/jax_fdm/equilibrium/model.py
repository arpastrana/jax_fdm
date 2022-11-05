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

    @classmethod
    def from_network(cls, network):
        """
        Create an equilibrium model from a force density network.
        """
        return cls(EquilibriumStructure(network))

    @partial(jit, static_argnums=0)
    def edges_vectors(self, xyz):
        return self.structure.connectivity @ xyz

    @partial(jit, static_argnums=0)
    def edges_lengths(self, vectors):
        return jnp.linalg.norm(vectors, axis=1, keepdims=True)

    @partial(jit, static_argnums=0)
    def edges_forces(self, q, lengths):
        return jnp.reshape(q, (-1, 1)) * lengths

    @partial(jit, static_argnums=0)
    def nodes_residuals(self, q, loads, vectors):
        connectivity = self.structure.connectivity
        return loads - np.transpose(connectivity) @ jnp.diag(q) @ vectors

    @partial(jit, static_argnums=0)
    def nodes_free_positions(self, q, xyz_fixed, loads):
        # convenience shorthands
        free = self.structure.free_nodes
        # loads = self.loads
        # xyz_fixed = self.xyz_fixed

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
    def nodes_positions(self, xyz_free, xyz_fixed):
        # NOTE: free fixed indices sorted by enumeration
        # xyz_fixed = self.xyz_fixed
        indices = self.structure.freefixed_nodes
        return jnp.concatenate((xyz_free, xyz_fixed))[indices, :]

    @partial(jit, static_argnums=0)
    def __call__(self, q, xyz_fixed, loads):
        """
        Compute an equilibrium state using the force density method.
        """
        xyz_free = self.nodes_free_positions(q, xyz_fixed, loads)
        xyz_all = self.nodes_positions(xyz_free, xyz_fixed)
        vectors = self.edges_vectors(xyz_all)
        residuals = self.nodes_residuals(q, loads, vectors)
        lengths = self.edges_lengths(vectors)
        forces = self.edges_forces(q, lengths)

        return EquilibriumState(xyz=xyz_all,
                                residuals=residuals,
                                lengths=lengths,
                                forces=forces,
                                vectors=vectors,
                                loads=loads,
                                force_densities=q)

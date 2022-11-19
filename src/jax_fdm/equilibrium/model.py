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

    def edges_vectors(self, xyz):
        """
        Calculate the unnormalized edge directions.
        """
        return self.structure.connectivity @ xyz

    def edges_lengths(self, vectors):
        """
        Compute the length of the edges.
        """
        return jnp.linalg.norm(vectors, axis=1, keepdims=True)

    def edges_forces(self, q, lengths):
        """
        Calculate the force in the edges.
        """
        return jnp.reshape(q, (-1, 1)) * lengths

    def nodes_residuals(self, q, loads, vectors):
        """
        Compute the force at the anchor supports of the structure.
        """
        connectivity = self.structure.connectivity
        return loads - np.transpose(connectivity) @ jnp.diag(q) @ vectors

    def nodes_free_positions(self, q, xyz_fixed, loads):
        """
        Calculate the XYZ coordinates of the free nodes.
        """
        # shorthand
        free = self.structure.free_nodes
        c_fixed = self.structure.connectivity_fixed
        c_free = self.structure.connectivity_free

        # connectivity
        c_free_t = np.transpose(c_free)

        # solve equilibrium after solving a linear system of equations
        q_matrix = jnp.diag(q)
        A = c_free_t @ q_matrix @ c_free
        b = loads[free, :] - c_free_t @ q_matrix @ c_fixed @ xyz_fixed

        return jnp.linalg.solve(A, b)

    def nodes_positions(self, xyz_free, xyz_fixed):
        """
        Concatenate in order the position of the free and the fixed nodes.
        """
        # NOTE: free fixed indices sorted by enumeration
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

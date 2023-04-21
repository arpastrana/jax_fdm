import jax

import equinox as eqx

from jax import vmap

import jax.numpy as jnp

from jax_fdm.equilibrium.state import EquilibriumState


# ==========================================================================
# Equilibrium model
# ==========================================================================


class EquilibriumModel(eqx.Module):
    """
    The equilibrium solver.
    """
    q: jax.Array
    xyz_fixed: jax.Array
    loads: jax.Array

    @classmethod
    def from_network(cls, network):
        """
        Create an equilibrium model from an force density network.
        """
        q, xyz_fixed, loads = (jnp.asarray(p) for p in network.parameters())

        return cls(q, xyz_fixed, loads)

    @staticmethod
    def edges_vector(structure, xyz):
        """
        Calculate the unnormalized edge directions.
        """
        return structure.connectivity @ xyz

    @staticmethod
    def edges_length(vectors):
        """
        Compute the length of the edges.
        """
        return jnp.linalg.norm(vectors, axis=1, keepdims=True)

    @staticmethod
    def edges_force(q, lengths):
        """
        Calculate the force in the edges.
        """
        return jnp.reshape(q, (-1, 1)) * lengths

    @staticmethod
    def nodes_residual(structure, q, loads, vectors):
        """
        Compute the force at the anchor supports of the structure.
        """
        connectivity = structure.connectivity
        return loads - connectivity.T @ jnp.diag(q) @ vectors

    @staticmethod
    def nodes_free_position(structure, q, xyz_fixed, loads):
        """
        Calculate the XYZ coordinates of the free nodes.
        """
        # shorthand
        free = structure.free_nodes
        c_fixed = structure.connectivity_fixed
        c_free = structure.connectivity_free

        A = c_free.T @ vmap(jnp.dot)(q, c_free)
        b = loads[free, :] - c_free.T @ vmap(jnp.dot)(q, c_fixed @ xyz_fixed)

        return jnp.linalg.solve(A, b)

    @staticmethod
    def nodes_position(structure, xyz_free, xyz_fixed):
        """
        Concatenate in order the position of the free and the fixed nodes.
        """
        # NOTE: free fixed indices sorted by enumeration
        indices = structure.freefixed_nodes
        return jnp.concatenate((xyz_free, xyz_fixed))[indices, :]

    def equilibrium(self, q, xyz_fixed, loads, structure):
        """
        Calculate a state of static equilibrium on a structure.
        """
        xyz_free = self.nodes_free_position(structure, q, xyz_fixed, loads)
        xyz_all = self.nodes_position(structure, xyz_free, xyz_fixed)
        vectors = self.edges_vector(structure, xyz_all)
        residuals = self.nodes_residual(structure, q, loads, vectors)
        lengths = self.edges_length(vectors)
        forces = self.edges_force(q, lengths)

        return EquilibriumState(xyz=xyz_all,
                                residuals=residuals,
                                lengths=lengths,
                                forces=forces,
                                vectors=vectors,
                                loads=loads,
                                force_densities=q)

    def __call__(self, structure):
        """
        Compute an equilibrium state using the force density method.
        """
        q = self.q
        xyz_fixed = self.xyz_fixed
        loads = self.loads

        return self.equilibrium(q, xyz_fixed, loads, structure)
